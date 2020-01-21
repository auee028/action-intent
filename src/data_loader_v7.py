import os
import json
import cv2
import numpy as np
import collections
import random
import nltk
import re
import time

import torch
import torchtext
import torchvision
from torch.utils.data import Dataset, DataLoader

from config_v7 import FLAGS


def file(json_file):
    with open(json_file) as f:
        return json.load(f)

def regx_process(sent):
    # Regular expressions used to tokenize.
    _WORD_SPLIT = re.compile("([·.,!?\"':;)(])")
    words = []
    for word in sent.strip().split():
        words.extend(_WORD_SPLIT.split(word))

    return ' '.join([w for w in words if w])

# prepare dataset and load mini-batch through collate_fn
class FrameDataset(Dataset):
    def __init__(self, mode, dataset):
        self.mode = mode            # (train, val, test)
        self.dataset = dataset      # (train, val_1, val_2)
        self.video_dir = os.path.join(FLAGS.video_prefix, dataset)
        self.sampled_dir = os.path.join(FLAGS.sampled_prefix, dataset)
        # self.activity200_dir = os.path.join(FLAGS.annotation_prefix, 'activitynet200.json')
        self.annotation_dir = os.path.join(FLAGS.annotation_prefix, '{}_short_data.json'.format(dataset))

        # self.class_index = self.get_classes()
        self.event_data = self.get_events()
        self.num_events = FLAGS.num_events
        self.event_list = list(self.event_data.keys())
        if mode == "train":
            random.shuffle(self.event_list)
        else:
            self.event_list.sort()

        self.start = 0
        self.selected_events = []

        self.frames_per_clip = FLAGS.frames_per_clip
        self.resize_short = FLAGS.resize_short
        self.crop_size = FLAGS.crop_size

        self.trg_max_seq_len = FLAGS.trg_max_seq_len
        self.embedding_size = FLAGS.embedding_size
        self.word2ix = self.create_vocab()

    def get_events(self):
        with open(self.annotation_dir, 'r') as f:
            event_data = json.load(f)
        # print(len(json_data))
        return event_data

    def create_vocab(self):
        print('Creating vocab ...')
        all_json_files = map(lambda _type: os.path.join(FLAGS.annotation_prefix, _type) + '_short_data.json',
                             ['train', 'val_1'])  # both of train and val_1 (val_2 is for test)
        all_data = {}
        for f in all_json_files:  # first train.json, next val_1.json
            with open(f, 'rb') as fp:
                all_data.update(json.load(fp))  # updates all of contents in a dictionary
        all_data = collections.OrderedDict(all_data).items()  # all captions are updated in all_data dictionary

        all_string = ' '.join([caption['sentence'].strip().lower() for _, caption in all_data])
        freq = collections.Counter(nltk.word_tokenize(regx_process(all_string))).most_common()  # len = 10399
        word2ix = {'<PAD>': FLAGS.PAD, '<SOS>': FLAGS.SOS, '<EOS>': FLAGS.EOS, '<UNK>': FLAGS.UNK}
        word2ix.update(dict(zip(list(zip(*freq))[0], range(4, len(freq) + 4))))  # len = 10403(= 10399+4)

        # with open('word2ix.json', 'w') as f:
        #     json.dump(word2ix, f)
        print("Done !")

        return word2ix

    def _collate_fn(self):
        if self.start < len(self.event_list):
            self.selected_events = self.event_list[self.start:self.start + self.num_events]
        else:   # self.start >= len(self.event_list)
            if self.mode == "train":
                copy = self.event_list[:self.start]
                random.shuffle(copy)
                self.event_list = self.event_list[self.start:] + copy
                print("\nShuffle !\n")

                self.start = 0

            else:
                self.selected_events = self.event_list[self.start - FLAGS.num_events:]

        print("--------- start: {} ---------".format(self.start))
        # print("{} data preparing ...".format(FLAGS.num_events))
        # s_t = time.time()

        frames, sentences = [], []
        for cur_indicator in self.selected_events:       # process for a single event
            info = self.event_data[cur_indicator]
            # print(info)
            duration = round(info['timestamp'][1] - info['timestamp'][0])
            if (duration > 35) | (duration < 1): continue

            frames_path = os.path.join(self.sampled_dir, info['video'], cur_indicator + '.npy')
            x = np.load(frames_path)
            # # image normalization
            # x_min = x.min(axis=(1, 2), keepdims=True)
            # x_max = x.max(axis=(1, 2), keepdims=True)
            x = 2 * x.astype(np.float32) / 255 - 1       # x_min = 0, x_max = 255, N ~ (-1, 1) = N ~ (0, 2) - 1
            frame_shape = x[0].shape
            cnt = x.shape[0]           # the number of frames of an event
            if (cnt < self.frames_per_clip):
                x = np.vstack((x,
                        np.zeros((self.frames_per_clip - cnt, ) + frame_shape)))  # pad with -30000 to diminish training of PAD)
            frames.append(x)


            sentences.append(self.get_tokenized_sentence(info['sentence']))


        src = np.array(frames)
        tar = sentences

        self.start += self.num_events

        # print("\t{} sec\n".format(time.time() - s_t))

        return (src, tar)

    def img_preprocess(self, frame, resize_shorter, crop_size):
        w, h = frame.shape[:2]

        if w < h:  # w is shorter
            new_w, new_h = (resize_shorter, int(resize_shorter * max(w, h) / min(w, h)))
        else:  # h is shorter
            new_w, new_h = (int(resize_shorter * max(w, h) / min(w, h)), resize_shorter)

        resized_img = cv2.resize(frame, dsize=(new_h, new_w),
                                 interpolation=cv2.INTER_AREA)  # caution: cv2.resize funtion needs the shape order of (h, w)

        center_w, center_h = (int(new_w / 2) - 1, int(new_h / 2) - 1)
        offset_w, offset_h = (int(crop_size[0] / 2), int(crop_size[1] / 2))
        cropped_img = resized_img[center_w - offset_w:center_w + offset_w, center_h - offset_h:center_h + offset_h]

        return cropped_img

    def get_tokenized_sentence(self, sentence):
        # embedded_sentence = [self.word2ix.get(word, FLAGS.UNK) \
        #                            for word in ['<SOS>']+nltk.word_tokenize(regx_process(sentence.strip().lower()))+['<EOS>']]
        tokenized_sentence = [self.word2ix.get(word, FLAGS.UNK) \
                                    for word in nltk.word_tokenize(regx_process(sentence.strip().lower()))]
        # if len(tokenized_sentence) <= (self.tar_max_seq_len-2):
        #     padded_sentence = [self.word2ix['<PAD>'] for _ in range(self.tar_max_seq_len-1)]
        #     padded_sentence[:len(tokenized_sentence)] = tokenized_sentence
        #     padded_sentence = [self.word2ix['<SOS>']] + padded_sentence + [self.word2ix['<EOS>']]
        #     # print(padded_sentence, len(padded_sentence))
        #     return padded_sentence
        # else:
        #     cropped_sentence = tokenized_sentence[:self.tar_max_seq_len-1]
        #     cropped_sentence = [self.word2ix['<SOS>']] + cropped_sentence + [self.word2ix['<EOS>']]
        #     # print(cropped_sentence, len(cropped_sentence))
        #     return cropped_sentence

        # print(sentence, np.shape(tokenized_sentence))

        return tokenized_sentence


class FeatsDataset(Dataset):
    def __init__(self, mode, dataset):
        self.mode = mode  # (train, val, test)
        self.dataset = dataset  # (train, val_1, val_2)
        self.feats_dir = os.path.join(FLAGS.feats_prefix, dataset)
        # self.activity200_dir = os.path.join(FLAGS.annotation_prefix, 'activitynet200.json')
        self.annotation_dir = os.path.join(FLAGS.annotation_prefix, '{}_new.json'.format(dataset))

        # self.class_index = self.get_classes()
        self.event_data = self.get_events()
        self.num_events = FLAGS.num_events
        self.event_list = list(self.event_data.keys())
        with open("missing_events", "r") as f:
            l = f.readlines()
            for el in l:
                if el.split('/')[0] == self.dataset:
                    self.event_list.remove(el.split('/')[1] + ' ' + el.split('/')[-1].split('.')[0].split('-')[-1])
                    print("Remove {}".format(el.split('/')[1] + ' ' + el.split('/')[-1].split('.')[0].split('-')[-1]))
        random.shuffle(self.event_list)

        sample_vid = os.path.join(self.feats_dir, os.listdir(self.feats_dir)[0])
        sample_event = np.load(os.path.join(sample_vid, os.listdir(sample_vid)[0]))
        self.feats_dim = sample_event.shape[-1]

        self.start = 0
        self.selected_events = []

        self.frames_per_clip = FLAGS.frames_per_clip
        self.resize_short = FLAGS.resize_short
        self.crop_size = FLAGS.crop_size

        self.trg_max_seq_len = FLAGS.trg_max_seq_len
        self.embedding_size = FLAGS.embedding_size
        self.word2ix = self.create_vocab()

    def get_events(self):
        with open(self.annotation_dir, 'r') as f:
            event_data = json.load(f)
        # print(len(json_data))
        return event_data

    def create_vocab(self):
        print('Creating vocab ...')
        all_json_files = map(lambda _type: os.path.join(FLAGS.annotation_prefix, _type) + '_new.json',
                             ['train', 'val_1'])  # both of train and val_1
        all_data = {}
        for f in all_json_files:  # first train.json, next val_1.json
            with open(f, 'rb') as fp:
                all_data.update(json.load(fp))  # updates all of contents in a dictionary
        all_data = collections.OrderedDict(all_data).items()  # all captions are updated in all_data dictionary

        all_string = ' '.join([caption['sentence'].strip().lower() for _, caption in all_data])
        freq = collections.Counter(nltk.word_tokenize(regx_process(all_string))).most_common()  # len = 10399
        word2ix = {'<PAD>': FLAGS.PAD, '<SOS>': FLAGS.SOS, '<EOS>': FLAGS.EOS, '<UNK>': FLAGS.UNK}
        word2ix.update(dict(zip(list(zip(*freq))[0], range(4, len(freq) + 4))))  # len = 10403(= 10399+4)

        # with open('word2ix.json', 'w') as f:
        #     json.dump(word2ix, f)
        print("Done !")

        return word2ix

    def _collate_fn(self):
        if self.start >= len(self.event_list):
            copy = self.event_list[:self.start]
            random.shuffle(copy)
            self.event_list = self.event_list[self.start:] + copy
            print("\nShuffle !\n")

            self.start = 0

        print("--------- start: {} ---------".format(self.start))

        # print("{} data preparing ...".format(FLAGS.num_events))
        # s_t = time.time()

        frames, sentences = [], []
        self.selected_events = self.event_list[self.start:self.start + self.num_events]
        for cur_indicator in self.selected_events:  # process for a single event
            info = self.event_data[cur_indicator]
            # print(info)

            feats_path = os.path.join(self.feats_dir, info['video'], 'event-' + cur_indicator.split(' ')[-1] + '.npy')
            x = np.squeeze(np.load(feats_path), axis=0)
            """
            # # image normalization
            # x_min = x.min(axis=(1, 2), keepdims=True)
            # x_max = x.max(axis=(1, 2), keepdims=True)
            x = 2 * x.astype(np.float32) / 255 - 1       # x_min = 0, x_max = 255, N ~ (-1, 1) = N ~ (0, 2) - 1
            """
            feats_len = x[0].shape
            cnt = x.shape[0]  # the number of frames of an event
            if (cnt < FLAGS.src_max_seq_len):
                x = np.vstack((x,
                               np.zeros((FLAGS.src_max_seq_len - cnt,) + feats_len)))
            else:
                x = x[:FLAGS.src_max_seq_len, :]
            frames.append(x)

            sentences.append(self.get_tokenized_sentence(info['sentence']))

        src = np.array(frames)
        tar = sentences

        self.start += self.num_events

        # print("\t{} sec\n".format(time.time() - s_t))

        return (src, tar)

    def img_preprocess(self, frame, resize_shorter, crop_size):
        w, h = frame.shape[:2]

        if w < h:  # w is shorter
            new_w, new_h = (resize_shorter, int(resize_shorter * max(w, h) / min(w, h)))
        else:  # h is shorter
            new_w, new_h = (int(resize_shorter * max(w, h) / min(w, h)), resize_shorter)

        resized_img = cv2.resize(frame, dsize=(new_h, new_w),
                                 interpolation=cv2.INTER_AREA)  # caution: cv2.resize funtion needs the shape order of (h, w)

        center_w, center_h = (int(new_w / 2) - 1, int(new_h / 2) - 1)
        offset_w, offset_h = (int(crop_size[0] / 2), int(crop_size[1] / 2))
        cropped_img = resized_img[center_w - offset_w:center_w + offset_w, center_h - offset_h:center_h + offset_h]

        return cropped_img

    def get_tokenized_sentence(self, sentence):
        # embedded_sentence = [self.word2ix.get(word, FLAGS.UNK) \
        #                            for word in ['<SOS>']+nltk.word_tokenize(regx_process(sentence.strip().lower()))+['<EOS>']]
        tokenized_sentence = [self.word2ix.get(word, FLAGS.UNK) \
                              for word in nltk.word_tokenize(regx_process(sentence.strip().lower()))]
        # if len(tokenized_sentence) <= (self.tar_max_seq_len-2):
        #     padded_sentence = [self.word2ix['<PAD>'] for _ in range(self.tar_max_seq_len-1)]
        #     padded_sentence[:len(tokenized_sentence)] = tokenized_sentence
        #     padded_sentence = [self.word2ix['<SOS>']] + padded_sentence + [self.word2ix['<EOS>']]
        #     # print(padded_sentence, len(padded_sentence))
        #     return padded_sentence
        # else:
        #     cropped_sentence = tokenized_sentence[:self.tar_max_seq_len-1]
        #     cropped_sentence = [self.word2ix['<SOS>']] + cropped_sentence + [self.word2ix['<EOS>']]
        #     # print(cropped_sentence, len(cropped_sentence))
        #     return cropped_sentence

        # print(sentence, np.shape(tokenized_sentence))

        return tokenized_sentence

    def avg_len(self):
        len_dict = {}
        avg = 0
        num_missing = 0

        for cur_indicator in self.event_list:
            vid_name = cur_indicator.split(' ')[0]

            feats_path = os.path.join(self.feats_dir, vid_name, 'event-' + cur_indicator.split(' ')[-1] + '.npy')
            try:
                x = np.squeeze(np.load(feats_path), axis=0)
                print(x.shape)
            except:
                num_missing += 1
                print(feats_path)
            """
            # # image normalization
            # x_min = x.min(axis=(1, 2), keepdims=True)
            # x_max = x.max(axis=(1, 2), keepdims=True)
            x = 2 * x.astype(np.float32) / 255 - 1       # x_min = 0, x_max = 255, N ~ (-1, 1) = N ~ (0, 2) - 1
            """
            feats_len = x.shape[0]
            if feats_len in len_dict:
                len_dict[feats_len] += 1
            else:
                len_dict[feats_len] = 1
        print(len_dict)

        for k, v in len_dict.items():
            avg += k * v
        avg /= (len(self.event_list) - num_missing)

        print("average length of feats : {}\n".format(avg))


def data_check():
    p = FLAGS.sampled_prefix

    for v in os.listdir(os.path.join(p, "train")):
        vl = os.listdir(os.path.join(p, "train", v))
        # print(vl)
        for vv in vl:
            x = np.load(os.path.join(p, "train", v, vv))
            if len(x) == 0:
                print(vv)


if __name__ == "__main__":
    mode = 'train'
    dataset = 'train'

    # frame_batcher = FrameDataset(mode=mode, dataset=dataset)
    feats_batcher = FeatsDataset(mode=mode, dataset=dataset)

    # print(frame_batcher.class_dict)
    # s, t = frame_batcher._collate_fn()
    # print(np.shape(s), np.shape(t))

    # data_check()

    feats_batcher.avg_len()