import tensorflow as tf
import numpy as np
import cv2, os
import model_zoo
import argparse
from optical_flow import detect_motion
from darknet.python.darknet import *
import copy


with open('action_map.txt', 'r') as f:
    action_labels = [line.strip() for line in f.readlines()]
with open('intent_map.txt', 'r') as f:
    intent_labels = [line.strip() for line in f.readlines()]
with open('demo_labels.txt', 'r') as f:
    demo_labels = [line.strip() for line in f.readlines()]

class TFModel:
    def __init__(self):
        parser = argparse.ArgumentParser(description="test TF on a single video")
        parser.add_argument('--video_length', type=int, default=64)
        parser.add_argument('--fps', type=int, default=15)
        parser.add_argument('--cam', type=int, default=0)
        parser.add_argument('--window_size', type=int, default=3)
        self.args = parser.parse_args()
        self.cap = cv2.VideoCapture(self.args.cam)
        self.cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, False)

        # internal counter
        self.cnt = 0

        # self.yolo = load_net("./darknet/cfg/yolov3-tiny-voc.cfg", "./darknet/cfg/yolov3-tiny-voc_210000.weights", 0)
        # self.meta = load_meta("./darknet/cfg/voc.data")

        self.yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
        self.meta = load_meta("./darknet/cfg/coco.data")

        g_action = tf.Graph()

        with g_action.as_default():
            self.videoclip_ph = tf.placeholder(dtype=tf.float32, shape=[1, 64, 224, 224, 3])

            # build entire pretrained networks (dummy operation!)
            net = model_zoo.I3DNet(inps=self.videoclip_ph,
                                   pretrained_model_path='../../pretrained/i3d-tensorflow/kinetics-i3d/data/kinetics_i3d/model',
                                   final_end_point='Predictions', trainable=False, scope='v/SenseTime_I3D')

            self.action_logits = net(inputs=self.videoclip_ph)

        g_intent = tf.Graph()
        with g_intent.as_default():
            self.action_labels_ph = tf.placeholder(dtype=tf.int32, shape=[1], name='action_labels_placeholder')

            # lstm for intent model
            num_units = 100
            dim_emb = 50
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
            lstm_state_vars = { 'c': tf.Variable(tf.zeros([1,num_units]), trainable=False),
                                'h': tf.Variable(tf.zeros([1,num_units]), trainable=False) }

            with tf.variable_scope('Intent'):
                # embedding layer
                embeddings = tf.get_variable('embeddings', shape=[400 + 1,
                                                                  dim_emb])
                action_emb = tf.nn.embedding_lookup(embeddings,
                                                    self.action_labels_ph)

                with tf.variable_scope('rnn'):
                    lstm_output, new_state = lstm_cell(action_emb,
                                                       tf.nn.rnn_cell.LSTMStateTuple(c=lstm_state_vars['c'],
                                                                                     h=lstm_state_vars['h']))

                n_intents = len(intent_labels)
                self.intent_logits = tf.layers.dense(lstm_output, n_intents, name='FC_intent')

            # update lstm state
            with tf.control_dependencies([lstm_output, new_state[0], new_state[1]]):
                self.update_state = [ lstm_state_vars['c'].assign(new_state[0]),
                                      lstm_state_vars['h'].assign(new_state[1]) ]

            self.init_state = [ lstm_state_vars['c'].assign(tf.zeros([1,num_units])),
                                lstm_state_vars['h'].assign(tf.zeros([1,num_units])) ]

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
        intent_saver.restore(self.sess_intent, save_path='./intent_model/model-last.ckpt')
        print("Restore intent model ...")


    def get_frames(self, num):

        seq = []
        while True:
            _, frame = self.cap.read()
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
            cv2.imwrite('./tmp/0.jpg',frame)

            boxes = detect(self.yolo, self.meta, './tmp/0.jpg')

            tmp = copy.copy(frame)

            # draw top-5 rect
            for box in boxes[:5]:
                lab, conf, (x,y,w,h) = box
                start = int(x-w/2),int(y-h/2)
                end = int(x+w/2),int(y+h/2)

                tmp = cv2.rectangle(tmp, start, end, (0,255,0), 1)
                cv2.putText(tmp, lab, start, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('rgb', tmp)
            cv2.moveWindow('rgb', 20, 20)
            cv2.waitKey(1)

            frame = cv2.resize(frame, (224, 224))

            seq.append(frame)

            if len(seq) >= num:
                return np.expand_dims(seq, axis=0)

    def init_LSTM_state(self):
        print('Init LSTM state ...')
        self.sess_intent.run(self.init_state)

    def run_demo_wrapper(self):
        if detect_motion(self.cap):
            frames = self.get_frames(num = self.args.video_length)

            action_logits = self.sess_act.run(self.action_logits,
                                              feed_dict={self.videoclip_ph: frames})

            # sort top five predictions from softmax output
            top_inds = action_logits[0].argsort()[::-1][:5]  # reverse sort and take five largest items
            if action_logits[0][top_inds[0]] < 0.2:
                return 'None', -1.0, 'None', -1.0

            valid_labels = list(filter(lambda x: action_labels[x+1] in demo_labels, top_inds))
            if not valid_labels:
                return 'None', -1.0, 'None', -1.0

            intent_logits, _ = self.sess_intent.run([self.intent_logits, self.update_state],
                                                    feed_dict={self.action_labels_ph: [valid_labels[0]+1]})

            self.cnt += 1
            if self.cnt % 5 == 0:
                self.init_LSTM_state()

            return action_labels[valid_labels[0]+1],\
                   action_logits[0].max(), \
                   intent_labels[intent_logits.argmax(axis=-1)[0]], \
                   intent_logits[0].max()

        else:
            self.init_LSTM_state()
            return 'None', -1.0, 'None', -1.0
