import cv2
import numpy as np
import multiprocessing

shared_q = multiprocessing.Queue()

def cam_routine(shared_q):
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    shared_q.put(img)

cam_process=multiprocessing.Process(target=cam_routine, args=(shared_q,))
cam_process.start()

while shared_q.empty():
    print('empty!!')

img_from_q = shared_q.get()
cv2.imshow('img_from_q', img_from_q)
cv2.waitKey(1)