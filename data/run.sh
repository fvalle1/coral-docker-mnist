#!/bin/sh

python3 /home/data/train.py
edgetpu_compiler -s -o /home/data/ /home/data/model.tflite
python3 /home/data/run.py
