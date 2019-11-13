# Dense captioning with I3D+Transformer model

### 1. Dataset
    1) I3D feature extraction: ActivityNet 200 (activitynet_200.json file consists of 'database', 'taxonomy', 'version' -> 'database' annotation is used)
    2) Transformer: ActivityNet Captions Dataset (train.json, val_1.json, val_2.json files)


### 2. Structure

 dense_captioning
    |--i3d
    |   |--i3d.py		: I3D model (https://github.com/rimchang/kinetics-i3d-Pytorch)
    |   |
    |--data
    |   |--annotation
    |   |	|__ activitynet200.json		: for classification (http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json)
    |   |	|				  'timestamp' is not used, but refered from '*_data_info.json'(Activitynet Captions dataset) file.
    |   |	|__ train.json			: for video captioning
    |   |	|__ val_1.json
    |   |	|__ val_2.json
    |   |	|__ train_ids.json
    |   |	|__ val_ids.json
    |   |	|__ test_ids.json
    |   |
    |   |
    |   |__ train01_16_128_171_mean.npy		: pre-trained weights
    |   |
    |   |__ train_data_info.json		: new annotation of train set
    |   |__ val_1_data_info.json		: new annotation of val_1 set
    |   |__ val_2_data_info.json		: new annotation of val_2 set 
    |
    |__ make_new_anno.py			: .py file to make new annotation
    |
    |__ data_loader.py				: .py file to load data in video_frames folder / return src=(B, 64, 224, 224, 3) and trg=(B, 16) / (B is seq_len for Transformer model.) / remove '#'s if you want to save word2ix.json
    |__ model.py				: I3d + embedding + positional encoding + Transformer + classifier + softmax
    |__ utils.py
    |__ config.py
    |__ main.py					: Run this.
    |
    |__ save_sampled_frames.py			: save sampled frames as .npy from .mp4 videos
    |__ check_save_sampled_frames.py		: check if sampled frames are saved as .npy
    |
    |__ (get_frames_by_event.py)		: .py file to extract and save .jpeg frames from .mp4 videos
    |__ (save_video_frames.py)			: failed because of scarse external memory space
    
 mydrive
    |--Dataset
    	|-- video		: a folder of .mp4 videos
    	|-- video_fraes		: a folder of extracted frames at fps=25 by event

### 3. Usage
    1) make_new_anno.py		: make new .json files(train_data_info.json, val_1_data_info.json, val_2_data_info.json). The files include annotation of {'vid event_idx': {'video':.., 'event_idx':.., 'timestamp':[start,end], 'sentence':..}, ...}
    2) save_video_frames.py	: save frames at fps=25 from .mp4 video files in 'video' folder to 'video_frames' folder by event (actually hard to use because of memory storage problem)


### 4. Requirements
python=3.6.1
tensorflow=1.14
tensorboard=1.14
pytorch=1.2
nltk


