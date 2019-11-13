import numpy as np
import cv2
import os
import json
import time


def get_events(annotation_dir):
    with open(annotation_dir, 'r') as f:
        event_data = json.load(f)
    # print(len(json_data))
    return event_data

def img_preprocess(frame, resize_short, crop_size):
    w, h = frame.shape[:2]

    if w < h:  # w is shorter
        new_w, new_h = (resize_short, int(resize_short * max(w, h) / min(w, h)))
    else:  # h is shorter
        new_w, new_h = (int(resize_short * max(w, h) / min(w, h)), resize_short)

    resized_img = cv2.resize(frame, dsize=(new_h, new_w),
                             interpolation=cv2.INTER_AREA)  # caution: cv2.resize funtion needs the shape order of (h, w)

    center_w, center_h = (int(new_w / 2) - 1, int(new_h / 2) - 1)
    offset_w, offset_h = (int(crop_size[0] / 2), int(crop_size[1] / 2))
    cropped_img = resized_img[center_w - offset_w:center_w + offset_w, center_h - offset_h:center_h + offset_h]

    return cropped_img

def get_frames_data(cap, timestamp, frames_per_clip, resize_short, crop_size):
    # provides one batch of 64 frames
    start, end = timestamp
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    sampling_rate = max(int(((end - start) * cap.get(cv2.CAP_PROP_FPS)) / frames_per_clip), 1)  # sampling by 4 frames
    # print("total number of frames: {}".format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    # print("sampling rate: {} frames".format(sampling_rate))

    event_frames = []
    error_msg = []

    cnt = 0
    cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while cap.get(cv2.CAP_PROP_POS_MSEC) < (end * 1000):
        ret, frame = cap.read()  # frame: numpy array
        if ret:
            # 64-frame max_seq_len-batch size append(padding or cropping)
            if cur_frame % sampling_rate == 0:
                processed_frame = img_preprocess(frame, resize_short, (crop_size, crop_size))
                event_frames.append(processed_frame)
                cnt += 1
                if cnt == frames_per_clip: break

            cur_frame += 1
        else:   # When cnt < frames_per_clip, it becomes complex.
            """
            # 'end' was rounded -> when actual end time is less than 'end' it causes infinite loop
            #   -> 'end' should be floored, not rounded
            #   -> But [timestamps] are already saved at json file ... -> Use break
            if cur_frame == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                # print(start, end)
                # print(cap.get(cv2.CAP_PROP_POS_MSEC), (end * 1000))
                msg = "end time in timestamps: {},  actual end time: {}".format(end*1000, cap.get(cv2.CAP_PROP_POS_MSEC))
                print("\t"+msg)
                error_msg.append(msg)
                break
            else:
                msg = "failed to read {}th/{} frame".format(cur_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                print("\t"+msg)
                error_msg.append(msg)
                cur_frame += 1
            """
            # So 'else:' is considered as the end of the mp4 video.
            break


        # if (len(event_frames) < frames_per_clip):
        #     for i in range(frames_per_clip - len(event_frames)):
        #         event_frames.append(np.zeros_like(event_frames[0]))  # pad with 0 becuase it's image
        # break

    # print(np.shape(event_frames))

    return event_frames, error_msg

if __name__=="__main__":
    mode = "train"      # train/val/test
    dataset = "train"   # train/val_1/val_2

    video_dir = os.path.join("/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/Dataset/Dense_VTT/video", mode)
    annotation_dir = os.path.join("./data", '{}_short_data.json'.format(dataset))
    save_dir = os.path.join("/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/Dataset/Dense_VTT/sampled_frames", mode)

    frames_per_clip = 64
    resize_short = 256
    crop_size = 224

    events_data = get_events(annotation_dir)
    events = list(events_data.keys())
    events.sort()

    num_total = len(events)
    num_frames = 0

    print("Start to extract frames (no padding) ...")
    for i, event in enumerate(events):
        start_time = time.time()

        frames = []

        # process for a single event
        info = events_data[event]
        # print(info)
        duration = round(info['timestamp'][1] - info['timestamp'][0])
        if (duration > 35) | (duration < 1): continue

        vid_path = os.path.join(video_dir, info['video'], info['video'] + '.mp4')
        frames_dir = os.path.join(save_dir, info['video'])

        if not os.path.exists(os.path.join(frames_dir, event+'.npy')):
            # print(vid_path)
            cap = cv2.VideoCapture(vid_path)
            # cap.set(cv2.CAP_PROP_FPS, FLAGS.fps)      # not set until opencv is compiled with ffmpeg
            # print("vid:{}, timestamp: {}, fps: {}".format(info['video'], info['timestamp'], cap.get(cv2.CAP_PROP_FPS)))
            # try:
            #     frames = get_frames_data(cap, info['timestamp'])
            #     print(type(frames))
            #     print("event: {} ,  frames: {}".format(event, np.shape(frames)[0]))
            #     num_frames += 1
            # except:
            #     with open("no_frames_list", "a") as f:
            #         f.write("event: {},  fps: {}".format(event, cap.get(cv2.CAP_PROP_FPS)))
            frames, _ = get_frames_data(cap, info['timestamp'], frames_per_clip, resize_short, crop_size)
            # print(type(frames))     # <class 'list'>
            """
            with open("alerts_list", "a") as f:
                f.write("{}\tevent: {} ,  frames shape: {} ,\n".format(i, event, np.shape(frames)))
                for msg in error_msg:
                    f.write("\tmsg: {}\n".format(msg))
            """

            if np.shape(frames)[0] != 0:
                if not os.path.exists(frames_dir):
                    os.makedirs(frames_dir)
                # if not os.path.exists(os.path.join(frames_dir, event+'.npy')):
                #     np.save(os.path.join(frames_dir, event), frames)
                #     print("{}\tevent: {} ,  frames: {} (shape: {})".format(i, event, np.shape(frames)[0], np.shape(frames)))
                #     num_frames += 1
                np.save(os.path.join(frames_dir, event), frames)
                print("{}\tevent: {} ,  frames: {} (shape: {})".format(i, event, np.shape(frames)[0], np.shape(frames)))
                num_frames += 1
            else:
                with open("alerts_list", "a") as f:
                    f.write("{}\tevent: {},  frames: {} (shape: {}),  fps: {}\n".format(i, event, np.shape(frames)[0], np.shape(frames), cap.get(cv2.CAP_PROP_FPS)))
        else:
            print("{}".format(i))

    print("total events: {}, frame-extracted events: {}".format(num_total, num_frames))
    with open("alerts_list", "a") as f:
        f.write("\ntotal events: {}, frame-extracted events: {}\n".format(num_total, num_frames))
