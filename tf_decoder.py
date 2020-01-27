import tensorflow as tf
from tensorflow.python.layers.core import Dense
from config import FLAGS

# embedding fn
_embedding_fn = lambda embedding,ids: tf.nn.embedding_lookup(embedding, ids)


def greedy_decoder(cell, embedding, initial_state, word_space_size,
                   dec_maxlen=40, dtype=tf.float32):
    # for test (greedy decoding)
    def initialize_fn():
        finished = tf.tile([False], [FLAGS.batch_size])
        start_inputs = _embedding_fn(embedding, tf.fill([FLAGS.batch_size], FLAGS.GO))
        return (finished, start_inputs)

    def sample_fn(time, outputs, state):
        """sample for CustomGreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, tf.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)

        return sample_ids

    def next_inputs_fn(time, outputs, state, sample_ids):
        """next_inputs_fn for CustomGreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = tf.equal(sample_ids, FLAGS.EOS)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: _embedding_fn(embedding, tf.fill([FLAGS.batch_size], FLAGS.GO)),
            lambda: _embedding_fn(embedding, sample_ids))

        return (finished, next_inputs, state)

    # custom greedy helper
    helper = tf.contrib.seq2seq.CustomHelper(initialize_fn=initialize_fn,
                                             sample_fn=sample_fn,
                                             next_inputs_fn=next_inputs_fn)

    # decoder projection
    decoder_projection_layer = Dense(word_space_size, use_bias=False, name='decoder_projection')

    # decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell, helper, initial_state=initial_state,
        output_layer=decoder_projection_layer)

    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                              maximum_iterations=dec_maxlen,
                                                              scope='rnn_dec/decoder')

    return decoder_outputs.rnn_output


def teacher_decoder(cell, inputs, embedding, initial_state, word_space_size,
                    sequence_length=None, dtype=tf.float32):

    if inputs is not None and sequence_length is None:
        sequence_length = tf.fill([FLAGS.batch_size], tf.shape(inputs)[1])

    helper = tf.contrib.seq2seq.TrainingHelper(_embedding_fn(embedding, inputs), sequence_length)

    # decoder projection
    decoder_projection_layer = Dense(word_space_size, use_bias=False, name='decoder_projection')

    # decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell, helper, initial_state=initial_state,
        output_layer=decoder_projection_layer)

    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                              scope='rnn_dec/decoder')

    return decoder_outputs.rnn_output
