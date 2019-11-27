import tensorflow as tf
import datetime

now = datetime.datetime.now()
tf.app.flags.DEFINE_string("version", '7', "trial version")

# training config
tf.app.flags.DEFINE_integer("num_events", 256, "number of event(only one event can be processed")
tf.app.flags.DEFINE_integer("fps", 25, "fps of input videos")
tf.app.flags.DEFINE_integer("frames_per_clip", 64, "frames per clip")
tf.app.flags.DEFINE_integer("batch_size", 1, "number of clips(max_sequence_length of an event for Transformer model)")
tf.app.flags.DEFINE_integer("epochs", 100, "number of epochs")
# tf.app.flags.DEFINE_integer("iterations", 200000, "iterations")
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "learning rate")
tf.app.flags.DEFINE_float("gamma", 0.1, "learning rate decay factor")
tf.app.flags.DEFINE_integer("lr_decay_step", 100000, "number of steps after which learning rate decays")
tf.app.flags.DEFINE_integer("resize_short", 256, "short length of resized image")
tf.app.flags.DEFINE_integer("crop_size", 224, "crop size")

# data config
tf.app.flags.DEFINE_string("feats_prefix", "/mnt/juhee/Dense_VTT/c3d_feats", "feats directory")
tf.app.flags.DEFINE_string("annotation_prefix", "./data", "directory of annotation in json file")
tf.app.flags.DEFINE_string("ckpt_dir", "/mnt/juhee/ckpt", "checkpoint directory")
tf.app.flags.DEFINE_string("idx_path", None, "path for loading i3d-feature-sampling-indices")
tf.app.flags.DEFINE_string("logs_dir", "/mnt/juhee/logs", "logs directory")
tf.app.flags.DEFINE_string("results_dir", "/mnt/juhee/results", "results directory")
tf.app.flags.DEFINE_string("writer_dir", "./tensorboard", "tensorboard writer directory")

# token config
tf.app.flags.DEFINE_integer("PAD", 0, "index of PAD symbol")
tf.app.flags.DEFINE_integer("SOS", 1, "index of GO symbol")
tf.app.flags.DEFINE_integer("EOS", 2, "index of EOS symbol")
tf.app.flags.DEFINE_integer("UNK", 3, "index of UNK symbol")

# model config
tf.app.flags.DEFINE_integer("i3d_dim", 1024, "i3d feature dimension")
tf.app.flags.DEFINE_integer("src_max_seq_len", 8, "max sequence length of source")   # avg length of frames(train) = 250
tf.app.flags.DEFINE_integer("trg_max_seq_len", 16, "max sequence length of target")   # avg length of captions(train) = 14.77
tf.app.flags.DEFINE_integer("embedding_size", 512, "embedding size")
tf.app.flags.DEFINE_integer("nhead", 8, "number of multi-heads")
tf.app.flags.DEFINE_integer("num_encoder_layers", 2, "number of encoder layers")
tf.app.flags.DEFINE_integer("num_decoder_layers", 6, "number of decoder layers")
tf.app.flags.DEFINE_integer("dim_feedforward", 512, "dimension of feedforward layer")
tf.app.flags.DEFINE_float("dropout", 0.1, "dropout rate")

# run config
tf.app.flags.DEFINE_string("mode", "train", "train/val/test")
tf.app.flags.DEFINE_string("dataset", "train", "train/val_1/val_2")

FLAGS = tf.app.flags.FLAGS
