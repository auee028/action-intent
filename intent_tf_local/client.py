import cv2
import numpy as np
import os, sys

# grpc things
from pbs import vintent_pb2, vintent_pb2_grpc
import grpc
import uuid
import sys
import requests
import copy
from optical_flow import detect_motion
import random

def normalize_probs(x):
    normalized = np.exp(x)/np.sum(np.exp(x))
    return normalized.tolist()

# interested objects
object_list = ['book',
                'bottle',
                'cell phone',
                'cup',
                'pizza',
                'orange',
                'banana',
                'mouse',
                'keyboard'
                'laptop',
                'person',
                'toothbrush', ]

class Visualizer:
    def __init__(self):
        self.init_box(-0.5, -0.5, -1, -1)

    def init_box(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def process(self, boxes, frame, frame_index):

        tmp = copy.copy(frame)

        person_positions = []

        # among the most significant bboxes
        for box in boxes[:5]:
            lab, conf, (x, y, w, h) = box
            start = int(x - w / 2), int(y - h / 2)
            end = int(x + w / 2), int(y + h / 2)
            if lab not in object_list:
                continue
            if lab == 'person':
                person_positions.append((x, y, w, h))
            if not lab == 'person':
                # draw rect and put text (except person)
                tmp = cv2.rectangle(tmp, start, end, (0, 255, 0), 1)
                cv2.putText(tmp, lab, start, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if not person_positions:
            return frame

        # person is detected!!
        ##

        mask_frame = 255*cv2.cvtColor(
            np.ones_like(tmp),
            cv2.COLOR_BGR2GRAY  # bgr -> rgb
        )

        big_PersonPos = person_positions[0]
        #if frame_index == 0:
        _x, _y, _w, _h = big_PersonPos
        self.init_box(_x, _y, _w, _h)

        x_low, x_max = int(max(0, self.x - self.w / 2)), \
                       int(min(mask_frame.shape[1], self.x + self.w / 2))
        y_low, y_max = int(max(0, self.y - self.h / 2)), \
                       int(min(mask_frame.shape[0], self.y + self.h / 2))

        mask_frame[y_low:y_max, x_low:x_max] = 255

        ### Resulting image(just for visualization)
        tmp = cv2.bitwise_and(tmp, tmp, mask=mask_frame)

        ####
        REGION = 100
        ####
        Rx_min, Rx_max = int(float(tmp.shape[1]) / 2 - REGION), int(float(tmp.shape[1]) / 2 + REGION)
        Ry_min, Ry_max = int(float(tmp.shape[0]) / 2 - REGION * 2), int(float(tmp.shape[0]) / 2 + REGION * 2)

        if Rx_min < self.x < Rx_max and \
                Ry_min < self.y < Ry_max:
            cv2.putText(tmp,
                        'Recognizing Action...',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255), 2)

        return tmp


class Client:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.cap = cv2.VideoCapture(int(sys.argv[1]))
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, False)

        self.action_hist = []; self.intent_hist = []
        self.object_hist = []

        self.visualizer = Visualizer()

    def generateRequests(self, frame, init):
        frame = frame.tobytes()
        yield vintent_pb2.MsgRequest(frame=frame, token=str(uuid.uuid4()), init=init)

    def send(self, frame, init):
        if init:
            print('init LSTM')
        channel = grpc.insecure_channel('{}:{}'.format(self.ip, self.port))
        stub = vintent_pb2_grpc.GreeterStub(channel)
        try:
            response = stub.Analyze(self.generateRequests(frame, init))
        except:
            pass
        return response

    def run(self):
        for i in range(64):
            ret, frame = self.cap.read()

            # response from server
            response = self.send(frame,False)

            self.object_hist.append(response.object)

            # roi_frame = self.visualizer.process(boxes=eval(response.boxes), frame=frame, frame_index=i)
            roi_frame = frame

            ret, jpeg = cv2.imencode('.jpg', roi_frame)
            stream = jpeg.tobytes()

            # for streaming
            requests.post('http://127.0.0.1:5001/update_stream', data=stream)

            if response.action_confidence>0.3:
                self.action_hist.append(response.action)
                self.intent_hist.append(eval(response.top3_intent_labels)[0])

        # print(self.action_hist, filter(lambda x: x!='None', self.intent_hist))
        top_action_counter = Counter(self.action_hist).most_common(1)
        top_object_counter = Counter(self.object_hist).most_common(1)
        top_intent_counter = Counter(self.intent_hist).most_common(1)

        top_action = 'Waiting...' if not top_action_counter else top_action_counter[0][0]
        top_object = 'Waiting...' if not top_object_counter else top_object_counter[0][0]
        top_intent = 'Waiting...' if not top_intent_counter or not top_action_counter else top_intent_counter[0][0]

        return top_action, top_object, top_intent, response.top3_intent_labels, response.top3_intent_probs


if __name__=='__main__':
    from collections import Counter
    client = Client(ip="localhost", port=50052)
    while True:
        top_action, top_object, top_intent, top3_intent_labels, top3_intent_probs = client.run()
        tmp_action = top_action
        tmp_intent = top_intent

        intent_panel = {"intent_labels": top3_intent_labels,
                        "intent_probs": str(normalize_probs(eval(top3_intent_probs)))}

        if tmp_action.startswith('eating') or tmp_action.startswith('tasting'):
            tmp_action = tmp_action.split()[0] + ' ' + 'something'
            if top_object == 'pizza' or top_object == 'banana' or top_object == 'orange':
                tmp_action = tmp_action.replace('something', top_object)
        elif tmp_action.startswith('drinking') or tmp_action.startswith('tasting'):
            tmp_action = 'drinking' + ' ' + 'something'
            if top_object =='cup' or top_object == 'bottle':
                tmp_action = tmp_action.split()[0] + ' ' + 'beverage with' + ' ' + top_object
        elif tmp_action.startswith('reading'):
            tmp_action = 'reading' + ' ' + 'something'
            if top_object == 'book':
                tmp_action = tmp_action.replace('something', top_object)
        elif tmp_action.startswith('juggling'):
            tmp_action = 'juggling'+ ' ' + 'something' if random.random()>0.5 else 'playing with'+ ' ' + 'something'
            if top_object == 'orange':
                tmp_action = tmp_action.replace('something', top_object)
        elif tmp_action=='fixing hair':
            tmp_action = "do one's hair"
        elif tmp_action=='rock scissors paper' or tmp_action=='shaking hands':
            tmp_action = "talking each other"

        # capitalize
        tmp_action = tmp_action.capitalize()
        tmp_intent = tmp_intent.capitalize()

        # send action label
        requests.get('http://127.0.0.1:5001/state/set/action',params={'action':tmp_action})
        # send action label to teamup chatbot controller
        requests.get('http://155.230.24.107:3003/chat/state/set/action',params={'action':'[ACTION],'+ tmp_action})

        requests.get('http://127.0.0.1:5001/state/set/intent_panel',
                     params=intent_panel)

        requests.get('http://127.0.0.1:5001/state/set/intent', params={'intent': tmp_intent})
        # send intent label to teamup chatbot controller
        #requests.get('http://155.230.24.107:3003/chat/state/set/intent',params={'intent':'[INTENT]/' + tmp_intent})

        print('Intent : {}'.format(tmp_intent))

        client.action_hist = []         # init action_hist
        client.object_hist = []         # init object_hist
        client.intent_hist = []         # init intent_hist
