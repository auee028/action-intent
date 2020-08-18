#!/usr/bin/env bash
# TODO. specify proto name(if any change)
source ~/.virtualenvs/context/bin/activate
python -m grpc_tools.protoc -I./src/protos --python_out=./src/pbs --grpc_python_out=./src/pbs ./src/protos/vintent.proto