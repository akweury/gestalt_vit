FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set the working directory inside the container
WORKDIR /app
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

#ARG GITHUB_TOKEN
RUN apt update && apt install -y git

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel
# Install Python dependencies with --no-cache-dir
WORKDIR /app

RUN pip install opencv-python==4.8.0.74
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install debugpy
RUN pip install pydevd-pycharm~=241.14494.241
