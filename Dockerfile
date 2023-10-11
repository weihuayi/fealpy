
FROM ubuntu:22.04

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && apt-get install -y ffmpeg

WORKDIR /data/fealpy
ADD requirements-full.txt ./
RUN pip install -r requirements-full.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# install application
COPY ./ ./
RUN pip install .

# configuration
EXPOSE 8000
