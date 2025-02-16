# 使用 Ubuntu 24.04 作为基础镜像
FROM ubuntu:24.04
# 更改 apt 镜像源为 Tsinghua 镜像源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \ 
    vim \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libscotch-dev \
    libmumps-scotch-5.6t64 \
    libmumps-scotch-dev \
    libmumps-ptscotch-5.6t64 \
    libmumps-ptscotch-dev \
    ffmpeg \
    && apt-get clean

RUN python3 -m venv /opt/fealpy-venv

# 激活虚拟环境并安装依赖
RUN /opt/fealpy-venv/bin/pip install --upgrade pip setuptools wheel

# 复制 requirements 文件并安装依赖
ADD requirements-full.txt ./
RUN /opt/fealpy-venv/bin/pip install --no-cache-dir -r requirements-full.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 PyTorch（CPU 版本）及其相关库
RUN /opt/fealpy-venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# 安装 FEALPy 和开发依赖
WORKDIR /data/fealpy


# 默认命令：启动 bash
CMD ["/bin/bash", "-c", "source /opt/fealpy-venv/bin/activate && exec /bin/bash"]

# configuration
EXPOSE 8888
