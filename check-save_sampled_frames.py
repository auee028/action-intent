import os
import glob
import numpy as np

path = '/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/Dataset/Dense_VTT/sampled_frames/train'
tr = 0
fa = 0
n = 0
l1 = glob.glob(path+"/**")
l1.sort()
for i in l1:
    l2 = glob.glob(i+'/*')
    l2.sort()
    for j in l2:
        n += 1
        e = np.load(j)
        print(n, j, np.shape(e))
        if np.shape(e)[0] == 0:
            fa += 1     # false : not extracted
        else:
            tr += 1     # true : extracted
print("\nx: {}, o: {}".format(fa, tr))
