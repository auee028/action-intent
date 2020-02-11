import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

with open('action_map.txt', 'r') as f:
    action_labels = [line.strip() for line in f.readlines()]
with open('intent_map.txt', 'r') as f:
    intent_labels = [line.strip() for line in f.readlines()]

batch_size = 100

def load_data(path):
    dataset = []
    for line in open(path,'r'):
        action_list, intent = line.strip().split('\t')
        action_list = list(map(lambda x: action_labels.index(x), action_list.split(',')))
        intent = intent_labels.index(intent)
        dataset.append({'action': action_list,
                        'intent': intent})
        # hi

    return dataset


def next_batch(dataset, batch_size):
    start = 0
    while True:
        if start > len(dataset)-batch_size:
            start = 0
            dataset = sorted(dataset, key=lambda x: np.random.rand())
            print('Shuffle...')

        slices = dataset[start:start+batch_size]
        maxlen = max(map(lambda x: len(x.get('action')), slices))
        cur_batch = {'action': map(lambda x: x.get('action')+[0]*(maxlen-len(x.get('action'))), slices),
                     'intent': map(lambda x: x.get('intent'), slices)}

        start += batch_size

        yield cur_batch

# build model
def build(n_actions, n_intents, dim_emb, num_units):
    action_labels = tf.placeholder(dtype=tf.int32, shape=[None,None], name='action_labels')
    intent_labels = tf.placeholder(dtype=tf.int32, shape=[None,None], name='intent_labels')

    with tf.variable_scope('Intent'):
        # embedding layer
        embeddings_act = tf.get_variable('embeddings_act', shape=[n_actions + 1, dim_emb])
        action_emb = tf.nn.embedding_lookup(embeddings_act,
                                            action_labels)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)

        lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=action_emb,
                                            sequence_length=tf.reduce_sum(tf.sign(action_labels),axis=1),
                                            dtype=tf.float32)

        embeddings_intent = tf.get_variable('embeddings_intent', shape=[n_intents + 1, dim_emb])

        lstm_outputs = tf.reshape(lstm_outputs, shape=[-1,dim_emb])

        intent_logits = tf.matmul(lstm_outputs,embeddings_intent,transpose_b=True)
        intent_logits = tf.reshape(intent_logits, shape=[batch_size, -1, n_intents+1])


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(intent_labels, n_intents+1),
                                                                logits=intent_logits))

    correct = tf.cast(tf.equal(intent_labels,
                       tf.cast(tf.argmax(intent_logits,axis=-1), tf.int32)), tf.float32)
    acc = tf.reduce_mean(correct)

    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

    return {
        'action_labels': action_labels,
        'intent_labels': intent_labels,
        'loss': loss,
        'train_op': train_op,
        'intent_logits': intent_logits,
        'acc': acc,
        'global_step': global_step
    }



def train():
    # train script
    m = build(n_actions=400,
              n_intents=10,
              dim_emb=50,
              num_units=50)

    print(tf.trainable_variables())

    sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=tf.trainable_variables())

    # load dataset & define generator for batching
    dataset = load_data(path='intent_dataset.txt')
    # dataset = load_data(path='/mnt/hdd1/intent_dataset-5.txt')

    # initial shuffling
    dataset = sorted(dataset, key=lambda x: np.random.rand())
    print('Inittial Shuffle...')

    gen = next_batch(dataset, batch_size=batch_size)

    history = {'loss': [],
               'acc': []}

    while True:
        try:
            b = next(gen)
            _action_labels = b['action']
            _intent_labels = b['intent']
            _masked_intent_labels = np.zeros_like(_action_labels).astype(np.float32)
            seq_lengths = np.sign(b['action']).sum(axis=1)
            for i in range(len(seq_lengths)):
                _masked_intent_labels[i, :seq_lengths[i]] = _intent_labels[i]

            acc, _, loss = sess.run([m['acc'], m['train_op'], m['loss']],
                                    feed_dict={m['action_labels']: _action_labels,
                                               m['intent_labels']: _masked_intent_labels})

            if m['global_step'].eval(sess) % 100 == 0:
                history['loss'].append(loss)
                history['acc'].append(acc)

                print('step : {}, loss: {}, acc : {}'.format(m['global_step'].eval(sess),
                                                             loss, acc))

        except KeyboardInterrupt:
            break

    plt.plot(range(len(history['loss'])), history['loss'], 'r',
             range(len(history['acc'])), history['acc'], 'b')
    plt.show()

    print('saving ckpt...')
    saver.save(sess,save_path='./intent_model/model-last.ckpt')

train()
