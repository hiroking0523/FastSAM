FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

RUN apt-get update
RUN apt-get install -y python3-tk

# pipでcv2がインストールされた場合に競合してしまうので無効化する
RUN mv /usr/lib/python3.8/dist-packages/cv2 /usr/lib/python3.8/dist-packages/cv2.bak
RUN mv /usr/local/lib/python3.8/dist-packages/cv2 /usr/local/lib/python3.8/dist-packages/cv2.

RUN pip install matplotlib pyyaml tqdm pandas psutil scipy seaborn gitpython
RUN pip install ultralytics