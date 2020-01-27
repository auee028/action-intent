import tensorflow as tf
from tensorflow.python.layers.core import Dense
from config import FLAGS
# TODO.
# 1) build short-term model, and train encoder/embeddings
# 2) build long-term model, and train it utilizing pre-trained encoder/embedding obtained from short-term model
# 3) note that when training long-term model, params of short-term encoder is freeze.(acts as feature extractor)
# 4) long-term model is composed of two multiple time scaled encoder, two outputs are concatenated and utilized as
#    initial state of long-term decoder

# model configurations
batch_size = FLAGS.batch_size
img_size = (FLAGS.height, FLAGS.width)
hidden_size = FLAGS.hidden_size
num_layers = FLAGS.num_layers
embedding_size = FLAGS.embedding_size
GO = FLAGS.GO
EOS = FLAGS.EOS
PAD = FLAGS.PAD

class C3DNet:
    def __init__(self, pretrained_model_path, scope=None, trainable=True):
        if scope == None:
            self.scope = 'C3D'

        # self.keep_rate = keep_rate

        with tf.variable_scope(self.scope):
            # load pre-trained weights(C3D)
            self._weights = {}
            self._biases = {}
            for var_name, var_shape in tf.contrib.framework.list_variables(pretrained_model_path):
                # load variable
                var = tf.contrib.framework.load_variable(pretrained_model_path, var_name)
                var_dict = self._biases if len(var_shape) == 1 else self._weights

                var_dict[var_name.split('/')[-1]] = tf.get_variable(var_name,
                                                                    var_shape,
                                                                    initializer=tf.constant_initializer(var),
                                                                    dtype='float32',
                                                                    trainable=trainable)

    def __call__(self, inputs):
        def conv3d(name, l_input, w, b):
            return tf.nn.bias_add(
                tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
                b, name=name)

        def max_pool(name, l_input, k):
            return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

        # Convolution Layer
        conv1 = conv3d('conv1', inputs, self._weights['wc1'], self._biases['bc1'])
        conv1 = tf.nn.relu(conv1, 'relu1')
        pool1 = max_pool('pool1', conv1, k=1)

        # Convolution Layer
        conv2 = conv3d('conv2', pool1, self._weights['wc2'], self._biases['bc2'])
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = max_pool('pool2', conv2, k=2)

        # Convolution Layer
        conv3 = conv3d('conv3a', pool2, self._weights['wc3a'], self._biases['bc3a'])
        conv3 = tf.nn.relu(conv3, 'relu3a')
        conv3 = conv3d('conv3b', conv3, self._weights['wc3b'], self._biases['bc3b'])
        conv3 = tf.nn.relu(conv3, 'relu3b')
        pool3 = max_pool('pool3', conv3, k=2)

        # Convolution Layer
        conv4 = conv3d('conv4a', pool3, self._weights['wc4a'], self._biases['bc4a'])
        conv4 = tf.nn.relu(conv4, 'relu4a')
        conv4 = conv3d('conv4b', conv4, self._weights['wc4b'], self._biases['bc4b'])
        conv4 = tf.nn.relu(conv4, 'relu4b')
        pool4 = max_pool('pool4', conv4, k=2)

        # Convolution Layer
        conv5 = conv3d('conv5a', pool4, self._weights['wc5a'], self._biases['bc5a'])
        conv5 = tf.nn.relu(conv5, 'relu5a')
        conv5 = conv3d('conv5b', conv5, self._weights['wc5b'], self._biases['bc5b'])
        conv5 = tf.nn.relu(conv5, 'relu5b')
        pool5 = max_pool('pool5', conv5, k=2)

        # Fully connected layer
        # pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3]) # only for ucf
        dense1 = tf.reshape(pool5, [batch_size, self._weights['wd1'].get_shape().as_list()[
            0]])  # Reshape conv3 output to fit dense layer input
        dense1 = tf.matmul(dense1, self._weights['wd1']) + self._biases['bd1']

        dense1 = tf.nn.relu(dense1, name='fc1')  # Relu activation
        #dense1 = tf.nn.dropout(dense1, self.keep_rate)

        dense2 = tf.nn.relu(tf.matmul(dense1, self._weights['wd2']) + self._biases['bd2'], name='fc2')  # Relu activation
        #dense2 = tf.nn.dropout(dense2, self.keep_rate)

        # Output: class prediction
        # out = tf.matmul(dense2, self._weights['wout']) + self._biases['bout']

        return dense1


def C3D_Captioner(inputs,
                  vocab_size,
                  keep_rate,
                  pretrained_model_path='../Models/C3D_VTT/pre-trained/sports1m_finetuning_ucf101.model',
                  captions=None,
                  sequence_length=None,
                  is_train=True,
                  dtype=tf.float32,
                  scope=None):
    """
    :param inputs: input frames, placeholder, float32, [0-255]
                   shape : [batch_size,time_depth,width,height,channels]
    :param vocab_size: vocab_size, int,
    :param keep_rate: keep rate for dropout, placeholder, float32, [0-1]
                   shape : [batch_size,time_depth,width,height,channels]

    :param captions: (optional in testing) ground-truth captions, placeholder, int32, [0-vocab_size]
                   shape : [batch_size,maxlen_sentence]
    :param sequence_length: (optional) length of each caption, tensor (or numpy array), int32, [0-maxlen_sentence]
                            For dynamic_rnn (decoder)
                            shape : [batch_size]
    :param is_train: is training mode?, boolean, default => True
    :param dtype: (optional) dtype, default => tf.float32
    :param scope: (optional) variable scope, default => "C3D_CAP"


    :return: outputs from each time step of decoder, Tensor
    """
    if scope==None:
        scope = 'C3D_CAP'

    def _Decoder(cnn_feats,
                 captions,
                 vocab_size,
                 sequence_length,
                 is_train,
                 keep_rate):

        # Word-embeddings
        with tf.device('/cpu:0'):
            embeddings = tf.get_variable('embeddings', [vocab_size, embedding_size])

        # Dynamic Decoder
        decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=keep_rate)
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell([decoder_cell]*num_layers)

        cnn_feats_embed = tf.layers.dense(cnn_feats, hidden_size, name='cnn_embed')

        with tf.variable_scope('decoder'):
            _, initial_state = decoder_cell(cnn_feats_embed, decoder_cell.zero_state(batch_size,'float32'))

        words = tf.concat([tf.fill([batch_size, 1], GO), captions[:, :-1]], 1)
        words_emb = tf.nn.embedding_lookup(embeddings, words)

        decoder_input = words_emb

        if is_train:
            # train helper (teaching for training mode)
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_input, sequence_length)
        else:
            # greedy embedding helper (greedy decoding mode)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, tf.fill([batch_size], GO), EOS)

        # decoder projection
        decoder_projection_layer = Dense(vocab_size, use_bias=False, name='decoder_projection')

        # decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper,
            initial_state=initial_state,
            output_layer=decoder_projection_layer)

        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                  maximum_iterations=None if is_train else 20,
                                                                  scope='decoder')

        return decoder_outputs.rnn_output

    with tf.variable_scope(scope,dtype=dtype):
        net = C3DNet(pretrained_model_path)
        c3d_feats = net(inputs=inputs)
        return _Decoder(c3d_feats,
                        captions, vocab_size, sequence_length,
                        is_train, keep_rate=keep_rate)

def Seq2Seq_encoder(inputs, keep_rate):
    def MTGRU(taus):
        cell = []
        for tau in taus:
            _cell = tf.nn.rnn_cell.MTGRUCell(num_units=hidden_size, tau=tau)
            _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=keep_rate)
            cell.append(_cell)

        cell = tf.nn.rnn_cell.MultiRNNCell(cell)
        return cell

    # Encoder (bidirectional)
    encoder_cell_fw = MTGRU(taus=[0.25, 0.5, 1.0])
    encoder_cell_bw = MTGRU(taus=[0.25, 0.5, 1.0])

    # RNN Encoder Network
    encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                                      cell_bw=encoder_cell_bw,
                                                                      inputs=inputs,
                                                                      dtype='float32',
                                                                      scope='encoder')

    return encoder_outputs, encoder_states

def Seq2Seq_decoder(captions,
                    embeddings,
                    vocab_size,
                    keep_rate,
                    sequence_length=None,
                    context_vector=None,
                    initial_state=None,
                    enable_attention=True, attention_states=None,
                    is_train=True, reuse=None):

    with tf.variable_scope('Decoder', reuse=reuse):
        # decoder cell with attention
        decoder_cell = []
        for _ in range(num_layers):
            _cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=keep_rate)
            decoder_cell.append(_cell)

        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell)

        if enable_attention:
            if attention_states is None:
                raise ValueError('Need attention_states to enable attention mechanism')

            # specify attention mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=hidden_size,
                                                                       memory=attention_states)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=hidden_size)  # decoder cell with attention

        if is_train:
            if captions is None:
                raise ValueError('Need captions to train seq2seq decoder')
            word_ids = tf.concat([tf.fill([batch_size, 1], GO),captions[:, :-1]], axis=1)
            decoder_inp = tf.nn.embedding_lookup(embeddings, word_ids)

            if sequence_length is None:
                sequence_length = tf.fill([batch_size],tf.shape(captions)[1])
            # train helper (teaching for training mode)
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_inp, sequence_length)
        else:
            # greedy embedding helper (greedy decoding mode)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, tf.fill([batch_size], GO), EOS)

        # decoder projection
        decoder_projection_layer = Dense(vocab_size, use_bias=False, name='decoder_projection')

        if initial_state is None:
            initial_state = decoder_cell.zero_state(dtype='float32', batch_size=batch_size)
        else:
            initial_state = initial_state

        # decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state=initial_state,
            output_layer=decoder_projection_layer)

        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                  maximum_iterations=None if is_train else 20)

        return decoder_outputs.rnn_output

def Seq2Seq_Captioner(inputs,
                      vocab_size,
                      keep_rate,
                      captions=None,
                      sequence_length=None,
                      is_train=True, enable_attention=True,
                      dtype=tf.float32,
                      scope=None):

    if scope==None:
        scope = 'Seq2Seq_Attn'


    with tf.variable_scope(scope, dtype=dtype):
        encoder_outputs, encoder_states = Seq2Seq_encoder(inputs, keep_rate)

        embeddings = tf.get_variable('embeddings', [vocab_size, embedding_size])

        return Seq2Seq_decoder(captions=captions,
                               embeddings=embeddings,
                               vocab_size=vocab_size,
                               sequence_length=sequence_length,
                               keep_rate=keep_rate,
                               initial_state=None,
                               enable_attention=enable_attention, attention_states=encoder_outputs,
                               is_train=is_train)


def S2VT_Captioner(inputs,
                   vocab_size,
                   keep_rate,
                   captions=None,
                   sequence_length=None,
                   is_train=True,
                   dtype=tf.float32,
                   scope=None):

    if scope==None:
        scope = 'S2VT'

    # visual rnn cell
    visual_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

    # language rnn cell
    language_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

    with tf.device('/cpu:0'):
        embeddings = tf.get_variable('embeddings', [vocab_size, embedding_size])

    def visual_encoder(inputs, keep_rate):
        # visual embedding
        visual_embedding = tf.layers.dense(inputs, hidden_size, name='visual_embedding')

        # visual (visual encoding phase)
        with tf.variable_scope('visual'):
            visual_encoder_outputs, visual_encoder_state = tf.nn.dynamic_rnn(cell=visual_cell,
                                                                             inputs=visual_embedding,
                                                                             dtype='float32')

        # concat (visual_encoder_outputs, <PAD> token embeddings)
        concat = tf.concat([visual_encoder_outputs,
                            tf.nn.embedding_lookup(embeddings,
                                tf.fill(dims=[batch_size, tf.shape(inputs)[1]], value=PAD))], axis=-1)

        # dropout!!
        concat = tf.nn.dropout(concat, keep_prob=keep_rate)

        # language (visual encoding phase)
        with tf.variable_scope('language'):
            _, language_encoder_state = tf.nn.dynamic_rnn(cell=language_cell,
                                                          inputs=concat,
                                                          dtype='float32')

        return visual_encoder_state, language_encoder_state

    def language_decoder(visual_encoder_state, language_encoder_state, keep_rate):
        # padding sequence for visual
        paddings = tf.zeros([batch_size, tf.shape(captions)[1], hidden_size])

        # visual (language decoding phase)
        with tf.variable_scope('visual', reuse=True):
            visual_decoder_outputs, _ = tf.nn.dynamic_rnn(cell=visual_cell,
                                                          inputs=paddings,
                                                          sequence_length=sequence_length,
                                                          initial_state=visual_encoder_state)

        ids = tf.concat([tf.fill([batch_size, 1], GO), captions[:,:-1]], 1)
        ids_emb = tf.nn.embedding_lookup(embeddings, ids)

        # concat (visual_decoder_outputs, caption embeddings)
        concat = tf.concat([visual_decoder_outputs, ids_emb], axis=-1)

        # dropout!!
        concat = tf.nn.dropout(concat, keep_prob=keep_rate)

        # language decoder (language decoding phase)
        with tf.variable_scope('visual', reuse=True):
            language_decoder_outputs, _ = tf.nn.dynamic_rnn(cell=language_cell,
                                                            inputs=concat,
                                                            sequence_length=sequence_length,
                                                            initial_state=language_encoder_state)

        logits = tf.layers.dense(language_decoder_outputs, vocab_size, name='logits')

        return logits

    def greedy_decoder(visual_encoder_state, language_encoder_state, maxlen=20):
        logits = []

        # padding for visual
        padding = tf.zeros([batch_size, hidden_size])

        # <go> symbol for captioning
        next_word = tf.fill([batch_size], GO)

        for t in range(maxlen):
            # visual encoder (language decoding phase)
            with tf.variable_scope('visual/rnn', reuse=t>0):
                visual_output, visual_encoder_state = visual_cell(padding, visual_encoder_state)

            word_emb = tf.nn.embedding_lookup(embeddings, next_word)
            concat = tf.concat([visual_output, word_emb], axis=-1)

            # language decoder (language decoding phase)
            with tf.variable_scope('language/rnn', reuse=t>0):
                language_output, language_encoder_state = language_cell(concat, language_encoder_state)

            cur_logits = tf.layers.dense(language_output, vocab_size, name='logits', reuse=t>0)
            next_word = tf.argmax(cur_logits, axis=-1)

            logits.append(cur_logits)

        return tf.transpose(logits, (1,0,2))

    with tf.variable_scope(scope,dtype=dtype):
        visual_encoder_state, language_encoder_state = visual_encoder(inputs=inputs,
                                                                      keep_rate=0.5 if is_train else 1.0)

        if is_train:
            return language_decoder(visual_encoder_state, language_encoder_state, keep_rate=keep_rate)
        else:
            return greedy_decoder(visual_encoder_state, language_encoder_state, maxlen=20)

class Dense_VideoCaptioner:
    def __init__(self,
                 C3D_feats,
                 start_end_time_stamp,
                 multi_word_id,
                 vocab_size,
                 keep_rate,
                 is_train=True):

        self.C3D_feats = C3D_feats
        self.start_end_time_stamp = start_end_time_stamp
        self.multi_word_id = multi_word_id
        self.vocab_size = vocab_size
        self.keep_rate = keep_rate

        # number of split
        self.n_split = tf.shape(self.start_end_time_stamp)[1]

        self.is_train = is_train

        self.embeddings = tf.get_variable('embeddings', [vocab_size, embedding_size])

    def encode(self):
        ### seq2seq encoder ###
        encoder_hiddens, state = Seq2Seq_encoder(inputs=self.C3D_feats,
                                                 keep_rate=self.keep_rate)

        return encoder_hiddens, state

    def get_context_vectors(self, encoder_hiddens):
        # start hiddens
        start_hiddens = tf.gather(encoder_hiddens, self.start_end_time_stamp[0,:,0], axis=1)
        # end hiddens
        end_hiddens = tf.gather(encoder_hiddens, self.start_end_time_stamp[0,:,1], axis=1)

        # concat of start&end hiddens
        context_vectors = tf.concat([start_hiddens,end_hiddens], axis=-1)

        return end_hiddens

    def decode(self, context_vectors=None, enable_attention=False, attention_states=None):
        # body function for while loop
        def compute_logits(n, logits_ta, reuse=True):
            # context = context_vectors[:,n]
            word_id = self.multi_word_id[:,n] if self.is_train else None

            #local_context = tf.layers.dense(context, num_layers*num_units, use_bias=False, name='context_embed', reuse=reuse)
            #local_context = tf.expand_dims(local_context, axis=1)   # for broadcasting
            #visual = tf.layers.dense(self.C3D_feats, num_layers*num_units, use_bias=False, name='visual_embed', reuse=reuse)

            # attention state (entire_context + local_context)
            # attention_states = tf.concat(encoder_hiddens, axis=-1) + local_context

            #init_state_tup = tuple([context] * num_layers)

            logits = Seq2Seq_decoder(captions=word_id,
                                     embeddings=self.embeddings,
                                     vocab_size=self.vocab_size,
                                     keep_rate=self.keep_rate,
                                     sequence_length=None,
                                     initial_state=None,
                                     enable_attention=enable_attention, attention_states=attention_states,
                                     is_train=self.is_train,
                                     reuse=reuse)

            return n+1, logits_ta.write(n, logits)

        n = tf.constant(0)
        logits_ta = tf.TensorArray(tf.float32, size=self.n_split)

        # initial usage
        n, logits_ta = compute_logits(n, logits_ta, reuse=False)

        # reuse all variables...
        _, logits_ta = tf.while_loop(lambda n, *_: n < self.n_split,
                                     body=compute_logits,
                                     loop_vars=(n, logits_ta))

        return tf.transpose(logits_ta.stack(),(1,0,2,3))

    def forward(self):
        encoder_hiddens, state = self.encode()
        context_vectors = self.get_context_vectors(encoder_hiddens)
        multi_logits = self.decode(context_vectors=context_vectors, enable_attention=True, attention_states=encoder_hiddens)
        return multi_logits