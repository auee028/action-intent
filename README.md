# CES_intent_demo


## Version info
Python : 2.7.12, CUDA : 8.x, CuDNN : 5.1.10

## Installation
Download C3D and I3D codes from github repositories, locate 'i3d.py' under the '/origin' and 'optical_flow.py'(opencv samples, https://github.com/opencv/opencv) under the '/origin' and 'intent_tf_local', and set the paths in the codes.
Download darknet codes from github repository, locate under the '/src/vintent'.

```
git clone https://github.com/hx173149/C3D-tensorflow.git
git clone https://github.com/deepmind/kinetics-i3d.git
git clone https://github.com/AlexeyAB/darknet
```

### TRAIN the model
```bash
cd ./orgin
python train_intent_model.py
```

### STEPS TO RUN THIS PROGRAM
```
# step 1 : run computation server (grpc)
cd ./src
python server.py

# step 2 : run flask app for web demo (just for html rendering)
cd ./intent_tf_local
python flask_app.py

# step 3 : run client.py('0' means cam index)
python client.py 0

# step 4 : visit 'localhost:5001' with Chrome
...
```

* If you want to build src/* as a new docker,
deploy/deploy.sh will do the whole process of 'build image->create & run container'
```
cd deploy
bash deploy.sh
```

* If you want to start already-existing container 'vintent_container',
```
nvidia-docker start vintent_container
``` 
