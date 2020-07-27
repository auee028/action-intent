# import cv2
# img = cv2.imread('/home/pjh/Pictures/flow.png', 0)
# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import os
import glob
import cv2
import numpy as np
import natsort

'''
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

# color images
writer = cv2.VideoWriter('/home/pjh/Videos/testVideo1.avi', fourcc, 30.0, (width, height))
# grayscale images
# writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height), 0)

while True:
    ret,img_color = cap.read()

    if ret == False:
        break

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Color", img_color)
    cv2.imshow("Gray", img_gray)

    writer.write(img_color)

    if cv2.waitKey(1)&0xFF == 27:   # Esc key
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
'''

"""
part_num = 15
homedir = '/home/pjh/Videos/test-arbitrary_parts/{}/*'.format(part_num)
sampledir = glob.glob(homedir)[0]

sample = cv2.imread(sampledir, 0)
height, width = sample.shape[:2]
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
writer = cv2.VideoWriter('/home/pjh/Videos/result-arbitrary_{}.avi'.format(part_num), fourcc, 5.0, (width, height))
# writer = cv2.VideoWriter('/media/pjh/FDAB-4AAC/test_0519-0226/video.avi', fourcc, 10.0, (width, height))

for frame_path in natsort.natsorted(glob.glob(homedir)):
    # print(frame_path)
    img = cv2.imread(frame_path)

    writer.write(img)

writer.release()
"""
def calc_framediff(root_path):
    # image_root = "/media/pjh/HDD2/Dataset/ces-demo-4th/ABR_action/1/2020-01-10-17-06-08_00_69/*"
    image_root = root_path + "/*"

    value_list = []
    for idx in range(len(glob.glob(image_root))-1):
        # print(image_dir)
        images = natsort.natsorted(glob.glob(image_root))

        prev_frame = cv2.imread(images[idx])
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.imread(images[idx+1])
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(prev_frame, curr_frame)
        frame_diff = np.array(frame_diff, dtype=np.int32)

        value_list.append(np.sum(frame_diff))

    # print(max(value_list), min(value_list))

    # scaling values to [0-1] range using formula
    for i in range(len(value_list)):
        value_list[i] = float(value_list[i] - np.min(value_list)) / (np.max(value_list) - np.min(value_list))

    # value_diff = sum(value_list)/len(value_list)
    # print(value_diff)
    # print(sum(value_list))
    return sum(value_list)

'''
# calculate each value of frame differences from class 1 to class 15 on one person
p = "/media/pjh/HDD2/Dataset/ces-demo-4th/ABR_action" + "/*"
val_list = []

for clss in natsort.natsorted(glob.glob(p)):
    first_person = natsort.natsorted(glob.glob(clss + "/*"))[59]
    print(glob.glob(first_person + "/*")[0])
    image = cv2.imread(glob.glob(first_person + "/*")[0], cv2.IMREAD_COLOR)
    cv2.imshow('image', image)
    cv2.waitKey(250)
    cv2.destroyAllWindows()

    sum_diff = calc_framediff(first_person)

    val_list.append(sum_diff)

    # print("{}\t{}".format(clss.split('/')[-1], sum_diff))

import matplotlib.pyplot as plt

num_clss = len(glob.glob(p))
x = range(1, num_clss+1)
y = val_list

plt.plot(x, y)
plt.show()

print('Avg. diff : {}'.format(sum(val_list) / num_clss))
'''

# calculate each value of frame differences from class 1 to class 15 on all data
p = "/media/pjh/HDD2/Dataset/ces-demo-4th/ABR_action" + "/*"
avg_list = [0.0 for _ in range(len(glob.glob(p)))]
# print(avg_list)

num_sample = 100
for clss in natsort.natsorted(glob.glob(p)):
    c = int(clss.split('/')[-1]) - 1
    # print(c)

    avg = 0.0
    for sample in glob.glob(clss + "/*")[:num_sample]:
        avg += calc_framediff(sample)

    avg_list[c] = avg / num_sample

    print("{}\t{}".format(c+1, avg_list[c]))

import matplotlib.pyplot as plt

num_clss = len(glob.glob(p))
x = range(1, num_clss+1)
y = avg_list

plt.plot(x, y)
plt.show()

print('Avg. diff : {}'.format(sum(avg_list) / num_clss))
