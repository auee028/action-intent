#!/usr/bin/env bash

tar -cvzf vintent.tar.gz ../src/*

nvidia-docker stop vintent_container
nvidia-docker rm vintent_container
nvidia-docker build --network host -t vintent_image .
rm -rf ./vintent.tar.gz
nvidia-docker run -d -p 50052:50052 --name vintent_container vintent_image
