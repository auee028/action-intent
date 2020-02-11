import numpy as np
import os, sys
import time
import grpc
from concurrent import futures
import cv2
import copy
import re

# grpc
from pbs import vintent_pb2, vintent_pb2_grpc

from vintent.TFModel import TFModel

# add to python path
intent_home = os.path.join(os.path.dirname(os.path.abspath(__file__)),'vintent')
sys.path.insert(0, intent_home)

from darknet.python.darknet import *

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

class TF_IntentModel:
    def __init__(self):
        self.tf_model = TFModel()
        self.video_stream = []

        self.action = 'None'
        self.top3_intent_labels = str(['None']*3)

        self.action_confidence = 0.0
        self.top3_intent_probs = str([0.0]*3)

    def run(self, frame, init):
        if init:
            print('Init LSTM state ...')
            self.tf_model.sess_intent.run(self.tf_model.init_state)
            self.video_stream = []
            return

        self.video_stream.append(frame)
        if len(self.video_stream) >= self.tf_model.video_length:
            action, action_confidence, top3_intent_labels, top3_intent_probs = self.tf_model.run(np.array([self.video_stream]))
            self.action = action
            self.top3_intent_labels = top3_intent_labels

            self.action_confidence = action_confidence
            self.top3_intent_probs = top3_intent_probs
            self.video_stream = []

def main():
    model = TF_IntentModel()

    class Greeter(vintent_pb2_grpc.GreeterServicer):
        def __init__(self):
            super(Greeter,self).__init__()

            # interested objects
            self.object_list = object_list

            self.init_box(-0.5,-0.5, -1, -1)

        def init_box(self, x, y, w, h):
            self.x, self.y, self.w, self.h = x, y, w, h

        def Analyze(self, request_iterator, context):
            for req in request_iterator:
                frame = np.frombuffer(req.frame, dtype=np.uint8).reshape(480, 640, 3)

                if not os.path.exists('./vintent/tmp'):
                    os.makedirs('./vintent/tmp')
                cv2.imwrite('./vintent/tmp/0.jpg', frame)

                boxes = detect(model.tf_model.yolo, model.tf_model.meta, './vintent/tmp/0.jpg')

                person_positions = []
                obj_positions = dict(names=[], positions=[])

                # among the most significant bboxes
                for box in boxes[:5]:
                    lab, conf, (x, y, w, h) = box
                    if lab not in self.object_list:
                        continue
                    if lab=='person':
                        person_positions.append((x,y,w,h))
                    else:
                        obj_positions['names'].append(lab)
                        obj_positions['positions'].append((x,y,w,h))

                if not person_positions:
                    print('no person!!!!!')
                    tmp_action = top_obj = 'None'
                    tmp_action_confidence = 0.0
                    tmp_top3_intent_labels = model.top3_intent_labels
                    tmp_top3_intent_probs = model.top3_intent_probs
                    break

                # person is detected!!
                ##


                big_PersonPos = person_positions[0]
                if not model.video_stream:
                    _x,_y,_w,_h = big_PersonPos
                    self.init_box(_x,_y,_w,_h)

                x_low, x_max = int(max(0, self.x - self.w / 2)), \
                               int(min(frame.shape[1], self.x + self.w / 2))
                y_low, y_max = int(max(0, self.y - self.h / 2)), \
                               int(min(frame.shape[0], self.y + self.h / 2))

                ####
                REGION = 100
                ####
                Rx_min, Rx_max = int(float(frame.shape[1])/2-REGION),int(float(frame.shape[1])/2+REGION)
                Ry_min, Ry_max = int(float(frame.shape[0])/2-REGION*2),int(float(frame.shape[0])/2+REGION*2)

                # lable with biggest confidence
                top_obj = 'None'
                person_center = big_PersonPos[:2]
                person_w, person_h = big_PersonPos[2:]

                for name, obj_pos in zip(obj_positions['names'], obj_positions['positions']):
                    x, y, w, h = obj_pos
                    obj_center = x,y

                    # compute distance between all objects and most significant person
                    if abs(person_center[0]-obj_center[0]) < (person_w + w)/2 and \
                            abs(person_center[1]-obj_center[1]) < (person_h + h)/2:
                        top_obj = name
                        break

                # if Rx_min < self.x < Rx_max and \
                #         Ry_min < self.y < Ry_max:
                # if most biggest person is centered at a center region,
                # then RUN DL model -> update action, intent status!
                # frame = cv2.resize(
                #    frame[y_low:y_max, x_low:x_max], (224, 224))

                # # center crop
                # frame_h, frame_w = frame.shape[:-1]
                # frame = frame[frame_h/2-112:frame_h/2+112,frame_w/2-112:frame_w/2+112,]
                #

                # force resizing
                frame = cv2.resize(frame, (224, 224))

                model.run(frame, req.init)

                tmp_action = model.action
                tmp_action_confidence = model.action_confidence

                tmp_top3_intent_labels = model.top3_intent_labels
                tmp_top3_intent_probs = model.top3_intent_probs

                model.action = 'None'
                model.action_confidence = 0.0

                # self.init_box(-0.5, -0.5, -1, -1)
                break

            return vintent_pb2.MsgReply(action=tmp_action,
                                   action_confidence=tmp_action_confidence,
                                   top3_intent_labels=tmp_top3_intent_labels,
                                   top3_intent_probs=tmp_top3_intent_probs,
                                   object=top_obj,
                                   boxes=str(boxes))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vintent_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)

if __name__=='__main__':
    main()
