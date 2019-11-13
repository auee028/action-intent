import os
import re
import json
import collections


def make_data_json(anno_file_path, save_file_path):
    data_info = collections.OrderedDict()

    with open(anno_file_path, 'r') as f:
        json_data = json.load(f)

    for vid, anno in collections.OrderedDict(json_data).items():
        timestamps = anno['timestamps']
        sentences = anno['sentences']
        for n, content in enumerate(zip(timestamps, sentences)):
            timestamp = content[0]
            if timestamp[0] > timestamp[1]:
                timestamp = [timestamp[1], timestamp[0]]
            sentence = content[1].strip()
            if sentence == "":
                print(vid, n)
            if (round(timestamp[1] - timestamp[0]) > 35) | (round(timestamp[1] - timestamp[0]) < 1):
                continue
            data_info[vid+' '+str(n)] = {'video':vid, 'event_idx':str(n), 'timestamp':timestamp, 'sentence':sentence}
    # print(data_info)
    print(len(data_info))   # num of events

    with open(save_file_path, 'w') as f:
        json.dump(data_info, f)

if __name__=="__main__":
    prefix_path = "/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/Dataset/Dense_VTT"
    mode = "train"  # (train, val_1, val_2) : (37421, 17502, 17028) events

    # video_prefix = os.path.join(prefix_path, 'video', mode)
    # frames_prefix = os.path.join(prefix_path, 'video_frames', mode)
    anno_file_path = os.path.join(prefix_path, "annotation/{}.json".format(mode))
    # save_file_path = "/home/pjh/PycharmProjects/dense-captioning/data/{}_data_info.json".format(mode)
    save_file_path = "/home/pjh/PycharmProjects/dense-captioning/data/{}_short_data.json".format(mode)

    make_data_json(anno_file_path, save_file_path)


