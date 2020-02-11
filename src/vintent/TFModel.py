import tensorflow as tf
import re
import os, sys
import numpy as np
import cv2
import requests

intent_home = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, intent_home)

import i3d
from darknet.python.darknet import *

class Normalizer:
    def __init__(self):
        self._inputs = tf.placeholder(tf.float32)

        # (min,max) => (-1.0,1.0)
        self._inputs_standardized = tf.image.per_image_standardization(self._inputs)
        self._inputs_normalized = tf.clip_by_value(self._inputs_standardized, -1.0, 1.0)


        # (min,max) => (-1.0,1.0)
        # self._inputs_normalized = (self._inputs-128.)/128

    def run(self, sess, input_data):
        return sess.run(self._inputs_normalized,
                        feed_dict={self._inputs:input_data})

class I3DNet:
    def __init__(self, inps, pretrained_model_path, final_end_point, trainable=False, scope='v/SenseTime_I3D'):

        self.final_end_point = final_end_point
        self.trainable = trainable
        self.scope = scope

        # build entire pretrained networks (dummy operation!)
        i3d.I3D(inps, scope=scope, is_training=trainable)

        var_dict = { re.sub(r':\d*','',v.name):v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='v/SenseTime_I3D') }
        self.assign_ops = []
        for var_name, var_shape in tf.contrib.framework.list_variables(pretrained_model_path):
            # load variable
            var = tf.contrib.framework.load_variable(pretrained_model_path, var_name)
            assign_op = var_dict[var_name].assign(var)
            self.assign_ops.append(assign_op)

    def __call__(self, inputs):
        out, _ = i3d.I3D(inputs, final_endpoint=self.final_end_point, scope=self.scope, is_training=self.trainable, reuse=True)
        return out


with open('./vintent/action_map.txt', 'r') as f:
    action_labels = [line.strip() for line in f.readlines()]
with open('./vintent/intent_map.txt', 'r') as f:
    intent_labels = [line.strip() for line in f.readlines()]
with open('./vintent/demo_labels.txt', 'r') as f:
    demo_labels = [line.strip() for line in f.readlines()]

class TFModel:
    def __init__(self, video_length=64):
        self.video_length = video_length

        # internal counter( used for reset signal )
        self.cnt = 0

        self.top_3_intent_labels = str(['None']*3)
        self.top_3_intent_probs = str([0.0]*3)

        # self.yolo = load_net("./vintent/darknet/cfg/yolov3-416.cfg", "./vintent/darknet/cfg/yolov3-416.weights", 0)
        # self.meta = load_meta("./vintent/darknet/cfg/coco.data")

        # load yolo v3(tiny) and meta data
        self.yolo = load_net("./vintent/darknet/cfg/yolov3-tiny-voc.cfg", "./vintent/darknet/cfg/yolov3-tiny-voc_210000.weights", 0)
        self.meta = load_meta("./vintent/darknet/cfg/voc.data")

        # self.yolo = load_net("./vintent/darknet/cfg/yolov3.cfg", "./vintent/darknet/cfg/yolov3.weights", 0)
        # self.meta = load_meta("./vintent/darknet/cfg/coco.data")

        g_action = tf.Graph()

        with g_action.as_default():
            self.videoclip_ph = tf.placeholder(dtype=tf.float32, shape=[1, None, 224, 224, 3])
            self.normalizer = Normalizer()

            # build entire pretrained networks (dummy operation!)
            net = I3DNet(inps=self.videoclip_ph,
                         pretrained_model_path='./vintent/kinetics_i3d/model',
                         final_end_point='Predictions', trainable=False, scope='v/SenseTime_I3D')

            self.action_logits = net(inputs=self.videoclip_ph)

        g_intent = tf.Graph()
        with g_intent.as_default():
            self.action_labels_ph = tf.placeholder(dtype=tf.int32, shape=[1], name='action_labels_placeholder')

            # lstm for intent model
            num_units = 50
            dim_emb = 50
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
            self.lstm_state_vars = { 'c': tf.Variable(tf.zeros([1,num_units]), trainable=False),
                                'h': tf.Variable(tf.zeros([1,num_units]), trainable=False) }

            with tf.variable_scope('Intent'):
                # embedding layer
                embeddings_act = tf.get_variable('embeddings_act', shape=[400 + 1,
                                                                  dim_emb])
                action_emb = tf.nn.embedding_lookup(embeddings_act,
                                                    self.action_labels_ph)

                with tf.variable_scope('rnn'):
                    lstm_output, new_state = lstm_cell(action_emb,
                                                       tf.nn.rnn_cell.LSTMStateTuple(c=self.lstm_state_vars['c'],
                                                                                     h=self.lstm_state_vars['h']))

                n_intents = len(intent_labels)
                embeddings_intent = tf.get_variable('embeddings_intent', shape=[n_intents, dim_emb])

                self.intent_logits = tf.nn.softmax(tf.matmul(lstm_output, embeddings_intent, transpose_b=True))

                # self.intent_logits = tf.nn.softmax(tf.layers.dense(lstm_output, n_intents, name='FC_intent'))

            # update lstm state
            with tf.control_dependencies([lstm_output, new_state[0], new_state[1]]):
                self.update_state = [ self.lstm_state_vars['c'].assign(new_state[0]),
                                      self.lstm_state_vars['h'].assign(new_state[1]) ]

            self.init_state = [ self.lstm_state_vars['c'].assign(tf.zeros([1,num_units])),
                                self.lstm_state_vars['h'].assign(tf.zeros([1,num_units])) ]

            # intent saver
            intent_saver = tf.train.Saver(tf.trainable_variables())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # action session
        self.sess_act = tf.Session(graph=g_action,
                                   config=config)
        # init all variables with pre-trained ckpt
        self.sess_act.run(net.assign_ops)

        # intent session
        self.sess_intent = tf.Session(graph=g_intent,
                                      config=config)
        with g_intent.as_default():
            self.sess_intent.run(tf.global_variables_initializer())

        # restore from trained intent-model ckpt
        print("Restore intent model ...")
        intent_saver.restore(self.sess_intent, save_path='./vintent/intent_model/model-last.ckpt')
        print("Done!")

    def run(self, frames):
        normalized_frames = []; batch_size = 1
        for ix in range(batch_size):
           cur_batch = []
           for _frame in frames[ix]:
               cur_batch.append(self.normalizer.run(
                   self.sess_act, _frame
               ))
           normalized_frames.append(cur_batch)

        action_logits = self.sess_act.run(self.action_logits,
                                          feed_dict={self.videoclip_ph: frames})\

        # sort top five predictions from softmax output
        top_ind = action_logits.argmax()  # reverse sort and take five largest items
        if action_logits[0][top_ind] < 0.3 or action_labels[top_ind+1] not in demo_labels:
            return 'None', 0.0, self.top_3_intent_labels, self.top_3_intent_probs

        intent_logits, _ = self.sess_intent.run([self.intent_logits, self.update_state],
                                                feed_dict={self.action_labels_ph: [top_ind+1]})
        # print('-----------update LSTM state...')

        self.cnt += 1
        if self.cnt % 5 == 0:
            print('Init LSTM state ...')
            self.sess_intent.run(self.init_state)

        _top_3_intent_labels = [ intent_labels[x+1] for x in intent_logits[0][1:].argsort()[::-1][:3] ]
        _top_3_intent_probs = np.array(intent_logits[0][1:][intent_logits[0][1:].argsort()[::-1]][:3])

        self.top_3_intent_labels = str(_top_3_intent_labels)
        self.top_3_intent_probs = str(list(_top_3_intent_probs))

        print(self.top_3_intent_labels, self.top_3_intent_probs)

        return action_labels[top_ind+1],\
               action_logits[0].max(), \
               self.top_3_intent_labels, \
               self.top_3_intent_probs
