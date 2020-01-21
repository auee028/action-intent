# -*- coding: utf-8 -*-
import numpy as np
import os
import json
import re
import collections
import matplotlib.pyplot as plt


def file(json_file):
    with open(json_file) as f:
        r = json.load(f)
    return r

# mode
mode = 'train'

# video features
feats_prefix = "/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/Dataset/Dense_VTT/video"
feats_dir = os.path.join(feats_prefix, mode)

# annotations
annotation_prefix = '/home/pjh/PycharmProjects/dense-captioning/data'
json_dir = os.path.join(annotation_prefix, mode + '_data_info.json')

json_data = collections.OrderedDict(file(json_dir)).items()  # output dictionary as saved order
cvt_data = list(json_data)  # convert type from OrederedDict to list
print("\ntotal video number(mode: {}) = {}\n".format(mode, np.shape(cvt_data)[0]))

durations = {}
for vid, anno in cvt_data:
    start, end = anno['timestamp']
    t = round(end - start)

    if t == 0: continue

    if t in durations:
        durations[t] = durations[t] + 1
    else:
        durations[t] = 1

with open("/home/pjh/PycharmProjects/dense-captioning/data/train_data_info.json", 'r') as f:
    import json
    data = json.load(f)

# average
sum = 0
for vid, info in data.items():
    t = round(info['timestamp'][1] - info['timestamp'][0])
    if t == 0: continue
    sum += t * durations[t]
avg = sum / len(data)
# print(avg)

# weighted average
w_avg = 0
for k, v in durations.items():
    if k == 0: continue
    w_avg += k * v
w_avg = w_avg / len(data)

# weighted average with the range of 1~35
maj_w_avg = 0
for k, v in durations.items():
    if (k < 1)|(k > 35): continue
    maj_w_avg += k * v
maj_w_avg = maj_w_avg / len(data)

print("--------------------------------------------------------")
print("weighted average of time duration: {}".format(w_avg))
print("weighted average of major time duration: {}".format(maj_w_avg))
print("number of durations: {}".format(len(sorted(durations.keys()))))
print("--------------------------------------------------------")

'''
hist_data = [0] * sorted(durations.keys())[-1]       # size : 245
len_range = [i+1 for i in range(len(hist_data))]
for k, v in durations.items():
    hist_data[k-1] = v
print(hist_data)

plt.figure(figsize=(60, 44))
plt.bar(len_range, hist_data)
# plt.hist(hist_data, bins=len(hist_data))
for i in range(0, len(hist_data)):
    plt.text(x=len_range[i]-1.9, y=hist_data[i]+5,
             s='{}'.format(hist_data[i]), ## 넣을 스트링
             fontsize=10,## 크기
             color='black',
             )
# x_min, x_max = plt.xlim() ## 글자가 안 보이는 경우가 있어서, 위의 길이를 조금 늘려줌
# y_min, y_max = plt.ylim() ## 글자가 안 보이는 경우가 있어서, 위의 길이를 조금 늘려줌
# plt.xlim(x_min, x_max+5)
# plt.ylim(y_min, y_max+100)
plt.title("TIME OF DURATIONS")
plt.xlabel("duration", labelpad=100)
plt.ylabel("frequency")
plt.xticks(len_range, rotation=-45)
# plt.yticks(range(0,max(list(seq_lens.values())),2000))
plt.gca().margins(x=0)
plt.savefig("duration_hist.png")
plt.show() # plt.show() have to be after plt.savefig() to save an image
'''