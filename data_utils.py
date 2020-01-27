#-*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import json
import re
import nltk
import random
import collections
import tensorflow as tf
from config import FLAGS

FRAMES_PER_CLIP = 16
SAMPLE_RATE = 10
CROP_SIZE = 112

def preprocess(frame, mean_frame):
    # mean frame from dataset
    mean_frame_height, mean_frame_width, _ = mean_frame.shape

    # mean subtract
    frame = cv2.resize(frame, (mean_frame_width,mean_frame_height))-mean_frame

    # center-crop, (w,h)=(112,112)
    res = frame[8:120, 30:142, :]

    return res

def update_embedding(filename, embedding, word2ix):
    file = open(filename,'r')
    update_ops = []
    for line in file.readlines():
        row = line.strip().split()
        word = row[0]

        ix = word2ix.get(word, None)
        if ix:
            update_ops.append(embedding[ix].assign(np.array(row[1:], dtype=np.float32)))

    file.close()

    return tf.group(*update_ops)


def regx_process(sent):
    # Regular expressions used to tokenize.
    _WORD_SPLIT = re.compile(b"([·.,!?\"':;)(])")
    words = []
    for word in sent.strip().split():
        words.extend(_WORD_SPLIT.split(word))

    return ' '.join([w for w in words if w])


class FrameBatcher:
    def __init__(self, type, batch_size=1,
                 annotation_prefix='./Dense_VTT/annotation'):
        self.type= type
        self.video_prefix = FLAGS.video_prefix
        self.annotation_prefix = annotation_prefix
        self.batch_size = batch_size
        self.json_path = os.path.join(annotation_prefix, type+'.json')
        self.data = collections.OrderedDict(json.load(file(self.json_path))).items()

        self.Batch = collections.namedtuple('Batch',['vid','frames','timestamps'])
        self.start = 0
        self.epoch = 0

    def prepare_feed_data(self, session, feats_sequence, tf_video_clip):
        _input_data_list = []

        batch = self.next_batch()

        _input_data_list.append(batch.vid)

        n_events = np.array(batch.timestamps).shape[1]

        # for broadcasting
        frame_arr = np.array(batch.frames[0])
        for j in range(n_events):
            try:
                cur_feats = session.run(feats_sequence,
                                        feed_dict={tf_video_clip: [frame_arr[j]]})
            except:
                # some data is trash...fuck
                print 'fuck'
                continue

            _input_data = cur_feats
            _input_data_list.append(_input_data)

        return _input_data_list

    def get_frames_data(self, cap, timestamps):
        crop_mean = np.load('train01_16_128_171_mean.npy').transpose(1, 2, 3, 0)  # (16, 128, 171, 3)
        frame_data = []

        for start, end in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            event_frames = []
            t = 0
            cnt = 0
            while cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000:
                ret, frame = cap.read()
                if ret:
                    if cnt % 10 == 0:
                        processed_frame = preprocess(frame, crop_mean[t % FRAMES_PER_CLIP])
                        event_frames.append(processed_frame)
                        t += 1
                    cnt += 1
                else:
                    break

            frame_data.append(event_frames)

        return frame_data

    def next_batch(self):
        cur_batch = self.Batch(vid=[], frames=[], timestamps=[])
        batch_sample = self.data[self.start:self.start+self.batch_size]

        for vid, anno in batch_sample:
            cap = cv2.VideoCapture(os.path.join(self.video_prefix, vid) + '.mp4')
            for ix, (cur_start, cur_end) in enumerate(anno['timestamps']):
                if cur_start > cur_end:
                    cur_start, cur_end = cur_end, cur_start

                anno['timestamps'][ix] = (cur_start, cur_end)

            frames = self.get_frames_data(cap, anno['timestamps'])

            # timestamps list of (start_ix,end_ix)
            cur_batch.timestamps.append(anno['timestamps'])

            cur_batch.vid.append(vid)
            cur_batch.frames.append(frames)

        self.start += self.batch_size

        return cur_batch

class FeatsBatcher:
    def __init__(self, type, batch_size=1,
                 annotation_prefix='./annotations'):
        self.type= type
        self.feats_dir = os.path.join(FLAGS.feats_home, type)
        self.annotation_prefix = annotation_prefix
        self.batch_size = batch_size
        self.json_path = os.path.join(annotation_prefix, type+'_demo.json')
        self.word2ix = self.create_vocab()
        self.data = collections.OrderedDict(json.load(file(self.json_path))).items()
        random.shuffle(self.data)
        print('Initial Shuffle !')

        self.Batch = collections.namedtuple('Batch', ['vid', 'feats', 'sentences', 'word_id'])
        self.start = 0
        self.epoch = 0

    def create_vocab(self):
        '''
        if not os.path.exists(self.json_path):
            print('Creating vocab ...')
            all_json_files = map(lambda _type: os.path.join(self.annotation_prefix,_type)+'_demo.json', ['train','val'])
            all_data = {}
            for f in all_json_files:
                with open(f,'rb') as fp:
                    all_data.update(json.load(fp))
            all_data = collections.OrderedDict(all_data).items()

            all_string = ' '.join([ cap.strip().lower() for _,anno in all_data for cap in anno['sentences'] ])
            freq = collections.Counter(nltk.word_tokenize(regx_process(all_string))).most_common()
            word2ix = {'<PAD>':0, '<GO>': 1, '<EOS>': 2, '<UNK>': 3}
            word2ix.update(dict(zip(zip(*freq)[0], range(4, len(freq) + 4))))

            json.dump(word2ix, file('word2ix_demo.json','w'))
            return word2ix
        else:
            print('Loading vocab ...')
            with open(self.json_path, 'r') as f:
                word2ix = json.load(f)
            return word2ix
        '''
        print('Loading vocab ...')
        with open(self.json_path, 'r') as f:
            word2ix = json.load(f)
        return word2ix

    def prepare_feed_data(self):
        _input_data_list = []
        _dec_in_list = []
        _dec_target_list = []

        batch = self.next_batch()

        _input_data_list.append(batch.vid)

        n_events = len(batch.feats)

        for j in range(n_events):
            cur_word_id = [FLAGS.GO] + batch.word_id[j][0]

            _input_data = batch.feats[j]
            _dec_in = cur_word_id[:-1]
            _dec_target = cur_word_id[1:]

            _input_data_list.append(_input_data)
            _dec_in_list.append([_dec_in])
            _dec_target_list.append([_dec_target])

        return _input_data_list, _dec_in_list, _dec_target_list

    def next_batch(self):
        if self.start > len(self.data)-self.batch_size:
            if self.type=='train':
                shuffled = sorted(self.data,key=lambda x:np.random.rand())
                self.data = collections.OrderedDict(shuffled).items()
            self.start = 0
            self.epoch += 1
            print 'End of Epoch... Shuffle !!'

        cur_batch = self.Batch(vid=[], feats=[], sentences=[], word_id=[])
        start = self.start# if self.type=='train' else np.random.randint(len(self.data)-self.batch_size)
        batch_sample = self.data[start:start+self.batch_size]

        feats_batch = []
        word_id_batch = [] # for padding multiple word_ids later ...
        for vid, anno in batch_sample:
            cur_feats = []
            for ix, event_id in enumerate(anno['sequence']):
                feats_path = os.path.join(self.feats_dir, vid, 'event-{}.npy'.format(ix))

                if os.path.exists(feats_path):
                    np_feats = np.load(feats_path).squeeze(axis=0)
                    cur_feats.append(np_feats)

            cur_batch.vid.append(vid)
            cur_batch.sentences.append([ ' '.join(nltk.word_tokenize(regx_process(s.strip().lower()))) for s in anno['sentences']])

            # feats
            feats_batch.append(cur_feats)

            # word_id
            word_id_batch.append([[self.word2ix.get(word, FLAGS.UNK) \
                                   for word in nltk.word_tokenize(regx_process(cap.strip().lower()))+['<EOS>'] ] \
                                  for cap in anno['sentences']])

        # feats padding
        max_split = max([len(feats_batch[i]) for i in range(self.batch_size)])
        max_len = max([feats_batch[i][j].shape[0] for i in range(self.batch_size) \
                       for j in range(max_split)])

        padded_feats = np.zeros([max_split, self.batch_size, max_len, FLAGS.input_size], dtype=np.float32)

        for i in range(max_split):
            for j in range(self.batch_size):
                padded_feats[i, j, :feats_batch[j][i].shape[0]] = feats_batch[j][i]

        cur_batch.feats.extend(padded_feats)

        # word padding
        max_split = max([len(word_id_batch[i]) for i in range(self.batch_size)])
        max_len = max([len(word_id_batch[i][j]) for i in range(self.batch_size) \
                       for j in range(max_split)])

        padded_word_id = np.zeros([max_split, self.batch_size, max_len], dtype=np.int32)

        for i in range(max_split):
            for j in range(self.batch_size):
                padded_word_id[i, j, :len(word_id_batch[j][i])] = word_id_batch[j][i]

        cur_batch.word_id.extend(padded_word_id.tolist())

        self.start += self.batch_size

        return cur_batch

if __name__=="__main__":
    batcher = FeatsBatcher(type="train", batch_size=1)

    x, y, z = batcher.prepare_feed_data()
    print(np.shape(x[0]), np.shape(y[0]), np.shape([0]))