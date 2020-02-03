import tensorflow as tf

# for multi rnn
tf.app.flags.DEFINE_integer("num_layers", 1, "num_layers")

# model config
tf.app.flags.DEFINE_integer("batch_size", 1, "batch size")
tf.app.flags.DEFINE_integer("input_size", 4096, "each input vector size")
tf.app.flags.DEFINE_integer("output_size", 1024, "each output vector size")
tf.app.flags.DEFINE_integer("words_count", 256, "words_count")
tf.app.flags.DEFINE_integer("word_size", 64, "word_size")
tf.app.flags.DEFINE_integer("read_heads", 4, "read_heads")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
tf.app.flags.DEFINE_integer("hidden_size", 1024, "hidden_size")
tf.app.flags.DEFINE_integer("embedding_size", 300, "embedding_size")
tf.app.flags.DEFINE_integer("dec_maxlen", 100, "dec_maxlen")

tf.app.flags.DEFINE_bool("with_context", True, "consider context or not")


tf.app.flags.DEFINE_integer("height", 112, "height of a frame")
tf.app.flags.DEFINE_integer("width", 112, "width of a frame")
tf.app.flags.DEFINE_integer("lr_decay_step", 60000, "learning rate decay step")
tf.app.flags.DEFINE_integer("iterations", 1000000, "number of iterations for training")
tf.app.flags.DEFINE_integer("start", 0, "number of iterations for training")
tf.app.flags.DEFINE_integer("stage", None, "stage index")

tf.app.flags.DEFINE_integer("FRAMES_PER_CLIP", 16, "frames per clip")

# token config
tf.app.flags.DEFINE_integer("PAD", 0, "index of GO symbol")
tf.app.flags.DEFINE_integer("GO", 1, "index of GO symbol")
tf.app.flags.DEFINE_integer("EOS", 2, "index of EOS symbol")
tf.app.flags.DEFINE_integer("UNK", 3, "index of UNK symbol")

# path
# tf.app.flags.DEFINE_string("mean_file", 'train01_16_128_171_mean.npy', "path to mean file from sports1m dataset")
# tf.app.flags.DEFINE_string("video_prefix", '/mnt/hdd1/Dataset/Dense_VTT/video_resize', "prefix of video")
tf.app.flags.DEFINE_string("feats_home", '/media/pjh/HDD2/Dataset/ces-demo-4th/demo_feats_balanced', "feats home dir")
tf.app.flags.DEFINE_string("pretrained_dir", '/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/SourceCodes/Dense_VTT/ckpt_repos/with_context_dnc_sports1m/step-60054', "pretrained weights directory")
tf.app.flags.DEFINE_string("checkpoint_dir", '/media/pjh/HDD2/Dataset/ces-demo-4th/ckpt/2020-02-02_16-54-47/step-54192', "checkpoint directory")
tf.app.flags.DEFINE_string("logs_dir", './logs', "logs directory")

# mode
tf.app.flags.DEFINE_string("mode", "eval", "support 'train/eval/demo' mode")

# for feature extraction
tf.app.flags.DEFINE_string("type", "train", "train/val/test")
# tf.app.flags.DEFINE_string("dir", None, "features target directory")

FLAGS = tf.app.flags.FLAGS