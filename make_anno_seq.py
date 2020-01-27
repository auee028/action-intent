import os
import glob
import random
import natsort
import re
import json
import tqdm
import numpy as np
import collections
import string

frames_root = "/media/pjh/HDD2/Dataset/ces-demo-4th/frames"
feats_root = "/media/pjh/HDD2/Dataset/ces-demo-4th/tmp_feats"

anno_list = "/media/pjh/HDD2/Dataset/ces-demo-4th/annotations/anno_list/{}.txt"
event_seq_list = "/media/pjh/HDD2/Dataset/ces-demo-4th/annotations/event_seq.txt"

def make_single_event_anno():
    # for dividing type(train/test)
    category_list = {}
    for i in range(1, 16):
        k = "class_{}".format(i)
        v = glob.glob(frames_root+"/*/{}/*".format(i))
        random.shuffle(v)

        category_list[k] = v

        # print(k, len(category_list[k]))

    trainset = []
    valset = []
    testset = []
    for k, v in category_list.items():
        N = len(v)

        trainset += v[:int(0.8*N)]
        valset += v[int(0.8*N):int(0.9*N)]
        testset += v[int(0.9*N):]
    print(len(trainset), len(valset), len(testset))

    new_anno = {}
    t = {'train':0, 'val':0, 'test':0}
    for v_path in glob.glob(frames_root+"/*/*/*"):
        video_name = v_path.split('/')[-1]
        category_num = v_path.split('/')[-2]
        person_name = v_path.split('/')[-3]

        dataset_type = 'train' if v_path in trainset else ''
        if dataset_type == '':
            dataset_type = 'val' if v_path in valset else 'test'

        t[dataset_type] += 1

        with open(anno_list.format(category_num), 'r') as f:
            caption_list = f.readlines()

        contents = collections.OrderedDict()
        contents['path'] =  v_path
        contents['person'] = person_name
        contents['category'] = category_num
        contents['type'] = dataset_type
        contents['sentence'] = random.sample(caption_list, 1)[0].strip()

        new_anno[video_name] = contents

    with open("./annotations/demo_annotation.json", "w") as f:
        json.dump(new_anno, f)
    print(t)

def rearrange_feats_dir():
    all_frames_dir = glob.glob(frames_root + "/*/*/*")
    print(len(all_frames_dir))
    for frames_dir in tqdm.tqdm(all_frames_dir):
        vid = frames_dir.split("/")[-1]

        feats = np.load(os.path.join(feats_root, vid, vid+".npy"))
        save_dir = frames_dir.replace("frames", "demo_feats")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, vid+".npy"), feats)

def make_event_seq_anno(save_root, feats_dir, sentence_dir):
    # load event sequences
    with open(event_seq_list, 'r') as f:
        seqs = f.readlines()

    # make annotation file with randomly selected sentences
    _LENGTH = 10    # 10 digits random string
    all_anno = {}
    anno_len = 0
    tmp_seq = []
    try:
        for _ in range(10):
            for seq_id, seq in enumerate(seqs):
                seq = seq.strip().split('-')

                # go through all people
                for person in os.listdir(frames_root):
                    # go through all dates
                    for date in ['2020-01-09', '2020-01-10', '2020-01-13']:
                        video_id = "demo_" + "".join([random.choice(string.ascii_letters) for _ in range(_LENGTH)])

                        if not os.path.exists(os.path.join(save_root, video_id)):
                            os.makedirs(os.path.join(save_root, video_id))

                        try:
                            pick_events = [random.choice(glob.glob(os.path.join(feats_dir, person, e, date+'*'))).split('/')[-1] for e in seq]
                        except:
                            continue
                        tmp_seq.append(pick_events)

                        # check a redundant sample
                        while pick_events not in tmp_seq:
                            pick_events = [random.choice(os.listdir(os.path.join(feats_dir, person, e))) for e in seq]

                        sentences = []
                        for n, event in enumerate(pick_events):
                            # np_feats = np.load(os.path.join(feats_dir, person, seq[n], event, event + '.npy'))
                            # print(np_feats.shape)
                            # np.save(os.path.join(save_root, video_id, 'event-{}.npy'.format(n)), np_feats)

                            with open(os.path.join(sentence_dir, "{}.txt".format(seq[n])), 'r') as f:
                                s_examples = f.readlines()
                            sentences.append(random.choice(s_examples).strip())

                        contents = collections.OrderedDict()

                        contents['person'] = person
                        contents['date'] = date
                        contents['seq_id'] = seq_id + 1
                        contents['sequence'] = seq
                        contents['orig_videos'] = pick_events
                        contents['sentences'] = sentences

                        all_anno[video_id] = contents

                        # anno_len += 1

            # if len(all_anno) == anno_len:
            #     break

        with open('annotations/demo_seq_anno.json', 'w') as f:
            json.dump(all_anno, f)

        print(len(all_anno))

    except:
        with open('annotations/demo_seq_anno.json', 'w') as f:
            json.dump(all_anno, f)

        print(len(all_anno))

    '''
    while True:
        for seq_id, seq in enumerate(seqs):
            seq = seq.strip().split('-')

            # go through all people
            cnt1 = 0
            for person in os.listdir(frames_root):

                video_id = "demo_" + "".join([random.choice(string.ascii_letters) for _ in range(_LENGTH)])

                if not os.path.exists(os.path.join(save_root, video_id)):
                    os.makedirs(os.path.join(save_root, video_id))

                pick_events = [random.choice(os.listdir(os.path.join(feats_dir, person, e))) for e in seq]
                tmp_seq.append(pick_events)

                # check a redundant sample
                cnt2 = 0
                while pick_events not in tmp_seq:
                    pick_events = [random.choice(os.listdir(os.path.join(feats_dir, person, e))) for e in seq]
                    cnt2 += 1

                    if cnt2 > 10:
                        print("break1")
                        break
                if cnt2 > 10:
                    cnt1 += 1
                    print("break2")
                    continue

                sentences = []
                for n, event in enumerate(pick_events):
                    np_feats = np.load(os.path.join(feats_dir, person, seq[n], event, event+'.npy'))
                    np.save(os.path.join(save_root, video_id, 'event-{}.npy'.format(n)), np_feats)

                    sentences.append(info[event]['sentence'])

                contents = collections.OrderedDict()

                contents['person'] = person
                contents['seq_id'] = seq_id
                contents['sequence'] = seq
                contents['orig_videos'] = pick_events
                contents['sentences'] = sentences

                all_anno[video_id] = contents

            if cnt1 > 20: break

        # with open('demo_seq_anno.json', 'w') as f:
        #     json.dump(all_anno, f)

        print(len(all_anno))
    '''

def divide_dataset(anno_file):
    with open(anno_file, 'r') as f:
        anno = json.load(f)

    train_set = {}
    val_set = {}
    test_set = {}
    for n, (k, v) in enumerate(anno.items()):
        if n < int(len(anno) * 0.8):
            train_set[k] = v
        elif (n >= int(len(anno) * 0.8)) & (n < int(len(anno) * 0.9)):
            val_set[k] = v
        else:
            test_set[k] = v
    print(len(train_set), len(val_set), len(test_set))  # (9032, 1129, 1129)

    with open("annotations/train_demo.json", 'w') as f:
        json.dump(train_set, f)
    with open("annotations/val_demo.json", 'w') as f:
        json.dump(val_set, f)
    with open("annotations/test_demo.json", 'w') as f:
        json.dump(test_set, f)

def save_feats_seq(anno_file, feats_dir, save_root):
    TYPE = anno_file.split('/')[-1].split('_demo')[0]
    with open(anno_file, 'r') as f:
        anno = json.load(f)

    for v_id, info in tqdm.tqdm(anno.items()):
        save_path = os.path.join(save_root, TYPE, v_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for n, event in enumerate(info['orig_videos']):
            np_feat = np.load(os.path.join(feats_dir, event, event+'.npy'))
            np.save(os.path.join(save_path, 'event-{}.npy'.format(n)), np_feat)


if __name__=="__main__":

    # rearrange_feats_dir()

    # STEP 1
    # make_single_event_anno()

    # STEP 2
    # save_root = "/media/pjh/HDD2/Dataset/ces-demo-4th/demo_feats"
    # feats_dir = "/media/pjh/HDD2/Dataset/ces-demo-4th/feats"
    # sentence_dir = "/media/pjh/HDD2/Dataset/ces-demo-4th/annotations/anno_list"
    # make_event_seq_anno(save_root=save_root, feats_dir=feats_dir, sentence_dir=sentence_dir)

    # STEP 3
    # anno_file = "annotations/demo_seq_anno.json"
    # divide_dataset(anno_file=anno_file)

    # STEP 4
    # anno_file = "annotations/{}_demo.json".format("test")       # train/val/test
    # feats_dir = "/media/pjh/HDD2/Dataset/ces-demo-4th/tmp_feats"
    # save_feats_seq(anno_file=anno_file, feats_dir=feats_dir, save_root=save_root)

    # STEP 5 (for check)
    with open("annotations/demo_seq_anno.json", 'r') as f:
        jfile = json.load(f)
    print(len(jfile))
    l = {2:0, 3:0, 4:0}
    for k, v in jfile.items():
        l[len(v['sentences'])] += 1
    print(l)