import tensorflow as tf
import numpy as np
import time
import sys, os
import json
import collections
import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from dnc.dnc import DNC
from dnc.recurrent_controller import RecurrentController
import tf_decoder #by hosang
from config import FLAGS
from data_utils import FeatsBatcher, update_embedding #by hosang
from utils import * #by hosang

# static vars
FRAMES_PER_CLIP = 16
SAMPLE_RATE = 10
CROP_SIZE = 112


# Define batchers
train_batcher = FeatsBatcher(type='train')
valid_batcher = FeatsBatcher(type='val')
test_batcher = FeatsBatcher(type=FLAGS.type)

# dict word->ix
word2ix = json.load(file('word2ix.json'))

# tf flags configurations
checkpoint_dir = FLAGS.checkpoint_dir
iterations = FLAGS.iterations

batch_size = FLAGS.batch_size
input_size = FLAGS.input_size
output_size = FLAGS.output_size
word_space_size = len(word2ix)
words_count = FLAGS.words_count
word_size = FLAGS.word_size
read_heads = FLAGS.read_heads

learning_rate = FLAGS.learning_rate

hidden_size = FLAGS.hidden_size         # decoder hidden size
embedding_size = FLAGS.embedding_size


def build_graph(ph, is_train):
    llprint("Building Computational Graph for DNC ...\n")

    with tf.variable_scope('DNC'):
        ncomputer = DNC(controller_class=RecurrentController,
                        input_size=input_size,
                        output_size=output_size,
                        keep_prob=ph.get('keep_prob', 1.0),
                        initial_memory_state=None,
                        memory_words_num=words_count,
                        memory_word_size=word_size,
                        memory_read_heads=read_heads,
                        batch_size=batch_size)

    with tf.variable_scope('embedding_layer'):
        embedding = tf.get_variable(name='embedding', shape=[word_space_size, embedding_size])

    # dec_cell
    dec_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)

    def loop_body(t, multi_dec_logits, memory_state):
        ncomputer.build_graph(input_data=tf.reverse(ph['multi_input_data'][t], axis=[1]), # reverse input data to reduce curse of padding
                              initial_memory_state=memory_state)
        outputs, memory_veiw = ncomputer.get_outputs()
        new_memory_state = memory_veiw['new_memory_state'] if FLAGS.with_context else ncomputer.memory.init_memory(None)

        initial_state = outputs[:, -1]

        if is_train:
            # build computation graph for decoder
            dec_logits = tf_decoder.teacher_decoder(cell=dec_cell,
                                                    inputs=ph['multi_dec_in'][t],
                                                    embedding=embedding,
                                                    initial_state=initial_state,
                                                    word_space_size=word_space_size,
                                                    sequence_length=None)
        else:
            dec_logits = tf_decoder.greedy_decoder(cell=dec_cell,
                                                   embedding=embedding,
                                                   initial_state=initial_state,
                                                   word_space_size=word_space_size,
                                                   dec_maxlen=FLAGS.dec_maxlen)
            # padded logits
            dec_logits = tf.pad(dec_logits,
                                [[0,0],[0,FLAGS.dec_maxlen-tf.shape(dec_logits)[1]],[0,0]])

        multi_dec_logits = multi_dec_logits.write(t, dec_logits)

        return t + 1, multi_dec_logits, new_memory_state

    memory_state = ncomputer.memory.init_memory(None)
    t = tf.constant(0)

    multi_dec_logits = tf.TensorArray(dtype=tf.float32, size=ph['n_events'])
    _, multi_dec_logits, _ = tf.while_loop(cond=lambda t,*_ : t<ph['n_events'],
                                           body=loop_body,
                                           loop_vars=(t, multi_dec_logits,memory_state))


    return dict(ncomputer=ncomputer,
                embedding=embedding,
                multi_dec_logits=multi_dec_logits.stack())

def build_train_graph():
    # placeholders only for decoder
    ph = dict(n_events=tf.placeholder(tf.int32, name='n_events'),
              multi_input_data=tf.placeholder(tf.float32, shape=[None,None,None,input_size], name='multi_input_data'),
              multi_dec_in=tf.placeholder(tf.int32, shape=[None,None, None], name='multi_dec_in'),
              multi_dec_target=tf.placeholder(tf.int32, shape=[None,None, None], name='multi_dec_target'),
              keep_prob=tf.placeholder(tf.float32, name='keep_prob'),
              )

    g = build_graph(ph, is_train=True)

    target_onehot = tf.one_hot(ph['multi_dec_target'], word_space_size)
    mask = tf.cast(tf.sign(ph['multi_dec_target']), tf.float32)

    crss_entropy_loss = mask*tf.nn.softmax_cross_entropy_with_logits(logits=g['multi_dec_logits'],
                                                                     labels=target_onehot)

    tot_loss = tf.reduce_sum(crss_entropy_loss)/(tf.reduce_sum(mask)+1e-12)

    global_step = tf.Variable(0, trainable=False)
    # learning rate decay
    starter_learning_rate = FLAGS.learning_rate # 0.1
    decay_steps = FLAGS.lr_decay_step
    decay_rate = FLAGS.lr_decay_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate,
                                               staircase=True)  # decay every decay_steps steps with a base of decay_rate

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(tot_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

    apply_gradients = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    train_fetches = [ g['multi_dec_logits'], tot_loss, apply_gradients ]
    valid_fetches = [ g['multi_dec_logits'], tot_loss, tf.no_op() ]

    return ph, train_fetches, valid_fetches

def build_test_graph():
    # placeholders only for decoder
    ph = dict(n_events=tf.placeholder(tf.int32, name='n_events'),
              multi_input_data=tf.placeholder(tf.float32, shape=[None,None,None,input_size], name='multi_input_data'),
              )

    g = build_graph(ph, is_train=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    llprint("Initializing Variables ... ")
    sess.run(tf.global_variables_initializer())
    llprint("Done!\n")

    # restore to continue training...
    if os.path.exists(checkpoint_dir):
        llprint("Restoring Checkpoint %s ... " % (checkpoint_dir))
        tf.train.Saver(tf.trainable_variables()).restore(sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        llprint("Done!\n")
    else:
        print("ckpt file is not found... \nreturn...")
        return

    return ph, g, sess

def step(sess, ph, fetches, batcher):
    feed_data = batcher.prepare_feed_data()

    multi_input_data, multi_dec_in, multi_dec_target = feed_data

    vid, _data_list = multi_input_data[0][0], multi_input_data[1:]

    feed_dict = {
        ph['n_events']: len(_data_list),
        ph['multi_input_data']: _data_list,
        ph['multi_dec_in']: multi_dec_in,
        ph['multi_dec_target']: multi_dec_target,
        ph['keep_prob']: 0.5 if batcher.type.startswith('train') else 1.0,
    }

    return dict(feed_results=sess.run(fetches, feed_dict=feed_dict),
                multi_dec_target=multi_dec_target)


def eval_step(sess, ph, fetches, batcher):
    feed_data = batcher.prepare_feed_data()

    multi_input_data, multi_dec_in, multi_dec_target = feed_data

    vid, _data_list = multi_input_data[0][0], multi_input_data[1:]

    feed_dict = {
        ph['n_events']: len(_data_list),
        ph['multi_input_data']: _data_list,
    }

    multi_dec_logits = sess.run(fetches, feed_dict=feed_dict)

    return vid, multi_dec_logits, multi_dec_target

def run(sess, ph, fetches, batcher, loss_list, score_list):
    # run training step
    results_dict = step(sess, ph,
                        fetches=fetches,
                        batcher=batcher)

    multi_dec_logits, multi_loss_values, _ = results_dict['feed_results']

    loss_list.append(
        np.mean(multi_loss_values)  # compute avg loss
    )

    score_list.append(
        calc_bleu(multi_dec_logits, results_dict['multi_dec_target'], word2ix)  # compute avg bleu score
    )


def train():
    # build computation graph on train phase
    ph, train_fetches, valid_fetches = build_train_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    llprint("Initializing Variables ... ")
    sess.run(tf.global_variables_initializer())
    llprint("Done!\n")

    # for log files
    now = datetime.datetime.now()
    logs_dir = os.path.join(FLAGS.logs_dir, now.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    train_logs = os.path.join(logs_dir, 'DNC_train.txt')
    val_logs = os.path.join(logs_dir, 'DNC_val.txt')
    configs = os.path.join(logs_dir, 'configs.txt')

    with open(configs, 'w') as f:
        f.writelines(["lr : {}\n".format(FLAGS.learning_rate),
                      "lr_decay_step : {}\n".format(FLAGS.lr_decay_step),
                      "lr_decay_rate : {}\n".format(FLAGS.lr_decay_rate),
                      "bs : {}\n\n".format(FLAGS.batch_size),
                      "feats_dir : {}\n".format(FLAGS.feats_home),
                      "anno_dir : {}\n".format("annotations/train_demo_{}.json".format(FLAGS.feats_home.split('_')[-1]))])

    last_100_losses = []; last_100_scores = []

    start = 0
    end = iterations + 1

    if os.path.exists(FLAGS.pretrained_dir):
        # restore to do fine tuning...
        pretrained_dir = FLAGS.pretrained_dir

        llprint("Restoring Checkpoint %s ... " % (pretrained_dir))
        tf.train.Saver(tf.trainable_variables()).restore(sess, os.path.join(pretrained_dir, 'model.ckpt'))
        llprint("Done!\n")

        # # restore to continue training...
        # ckpt_list = os.listdir(os.path.dirname(FLAGS.checkpoint_dir))
        # ckpt_list = filter(lambda x: x != 'step-last', ckpt_list)
        # ckpt_latest = os.path.join(os.path.dirname(FLAGS.checkpoint_dir),
        #                            sorted(ckpt_list, key=lambda x: int(x.split('-')[1]), reverse=True)[0])
        #
        # llprint("Restoring Checkpoint %s ... " % (ckpt_latest))
        # tf.train.Saver(tf.trainable_variables()).restore(sess, os.path.join(ckpt_latest, 'model.ckpt'))
        # llprint("Done!\n")
        #
        # # update start index
        # start = int(os.path.basename(pretrained_dir).split('-')[1]) + 1

    start_time_100 = time.time()
    avg_100_time = 0.
    avg_counter = 0
    while True:
        try:
            start += 1
            llprint("\rIteration %d/%d" % (start, end))

            summerize = (start % (100) == 0)
            take_checkpoint = (start % 5000 == 0) # (start % (len(train_batcher.data)/batch_size) == 0)
            validate = (start % 1000 == 0)

            # run training step
            run(sess=sess, ph=ph, fetches=train_fetches,
                batcher=train_batcher,
                loss_list=last_100_losses,
                score_list=last_100_scores)

            if validate:
                llprint("\rValidation for 100 samples at %d" % start)

                last_100_val_losses = []; last_100_val_scores = []
                while len(last_100_val_losses)<100:
                    # run validation step
                    run(sess=sess, ph=ph, fetches=valid_fetches,
                        batcher=valid_batcher,
                        loss_list=last_100_val_losses,
                        score_list=last_100_val_scores)

                llprint("\n\t[Validation] Avg. LOSS: %.7f\t Avg. BLEU: %.7f\n" % (float(np.mean(last_100_val_losses)),
                                                                                  float(np.mean(last_100_val_scores))))
                write_logs(fn=val_logs, step=start, loss=np.mean(last_100_val_losses), score=np.mean(last_100_val_scores))

            if summerize:
                llprint("\n\tAvg. LOSS: %.7f\t Avg. BLEU: %.7f\n" % (float(np.mean(last_100_losses)),
                                                                     float(np.mean(last_100_scores))))
                write_logs(fn=train_logs, step=start, loss=np.mean(last_100_losses), score=np.mean(last_100_scores))

                last_100_losses = []; last_100_scores = []

                end_time_100 = time.time()
                elapsed_time = (end_time_100 - start_time_100) / 60
                avg_counter += 1
                avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                estimated_time = (avg_100_time * ((end - start) / 100.)) / 60.

                print ("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                print ("\tApprox. time to completion: %.2f hours" % (estimated_time))

                start_time_100 = time.time()

            if take_checkpoint:
                llprint("\nSaving Checkpoint ... "),
                # save
                checkpoint_save_dir = checkpoint_dir.replace(os.path.basename(checkpoint_dir), 'step-{}'.format(start)).format(now.strftime('%Y-%m-%d_%H-%M-%S'))

                if not os.path.exists(checkpoint_save_dir):
                    os.makedirs(checkpoint_save_dir)

                tf.train.Saver(tf.trainable_variables()).save(sess, os.path.join(checkpoint_save_dir, 'model.ckpt'))

                llprint("Done!\n")

        except KeyboardInterrupt:
            llprint("\nSaving Checkpoint ... "),
            # save
            checkpoint_save_dir = checkpoint_dir.replace(os.path.basename(checkpoint_dir), 'step-last').format(now.strftime('%Y-%m-%d_%H-%M-%S'))

            if not os.path.exists(checkpoint_save_dir):
                os.makedirs(checkpoint_save_dir)

            tf.train.Saver(tf.trainable_variables()).save(sess, os.path.join(checkpoint_save_dir, 'model.ckpt'))
            llprint("Done!\n")

            time_info = "\nTime for training: {:.4f} hours".format((time.time() - start_time_100)/60/60)
            print(time_info)
            with open(configs, 'a') as f:
                f.write(time_info)

            sys.exit(0)

def eval():
    # build and restore computation graph on test phase
    ph, g, sess = build_test_graph()

    step_num = os.path.basename(checkpoint_dir).split('-')[1]

    # generate data for evaluation
    datasetGold, datasetHypo = generate_eval_data(sess, ph, g, step_func=eval_step, batcher=test_batcher, word2ix=word2ix)

    ckpt = FLAGS.checkpoint_dir.split('/')[-2]

    if FLAGS.with_context:
        result_dir = './results_with_context/{}'.format(ckpt)
    else:
        result_dir = './results_no_context/{}'.format(ckpt)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    json.dump(datasetGold, file(os.path.join(result_dir,'gold_{}-{}.json').format(FLAGS.type, step_num), 'wb'))
    json.dump(datasetHypo, file(os.path.join(result_dir,'hypo_{}-{}.json').format(FLAGS.type, step_num), 'wb'))


def demo():
    # build and restore computation graph on test phase
    ph, g, sess = build_test_graph()

    step_num = os.path.basename(checkpoint_dir).split('-')[1]

    if FLAGS.with_context:
        result_dir = './demo_with_context'
    else:
        result_dir = './demo_no_context'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    JSONDemo = generate_JSONDEMO(sess,ph,g,step_func=eval_step,batcher=valid_batcher, word2ix=word2ix)

    json.dump(JSONDemo, file(os.path.join(result_dir,'web_demo-{}.json').format(step_num), 'wb'))


if __name__=='__main__':
    if FLAGS.mode=='train':
        train()
    elif FLAGS.mode=='eval':
        eval()
    elif FLAGS.mode == 'demo':
        demo()