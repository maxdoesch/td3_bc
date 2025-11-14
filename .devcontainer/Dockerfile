FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

###############################################################################

#### Match User and Group id ####

ARG USERNAME=smlr
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG USER_HOME=/home/$USERNAME

# Delete existing user if it exists
RUN if getent passwd ${USER_UID}; then \
    userdel -r $(getent passwd ${USER_UID} | cut -d: -f1); \
    fi

# Delete existing group if it exists
RUN if getent group ${USER_GID}; then \
    groupdel $(getent group ${USER_GID} | cut -d: -f1); \
    fi

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

# Set up sudo
RUN apt-get update \
  && apt-get install -y sudo \
  && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

###############################################################################

#### Install essential packages ####

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates unzip curl git git-all nano vim tmux gcc g++ \
    build-essential cmake pkg-config software-properties-common apt-transport-https htop && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

###############################################################################

#### Python 3.10 ####

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel pytest pytest-cov flake8 && \
    rm -rf /var/lib/apt/lists/*

###############################################################################

#### Install EGL for Headless Rendering ####

RUN apt-get update && \
    apt-get install -y libegl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libopengl0 \
    libosmesa6 \
    libosmesa6-dev

###############################################################################

SHELL ["/bin/bash", "-c"]

ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh

###############################################################################

#### Install Python Packages Requirements ####

RUN python -m pip install gdown && \
    python -m pip install uv

###############################################################################

# Switch to non-root user
USER $USERNAME

#### Python Environment ####

# MobileSAMv2
WORKDIR $USER_HOME
RUN git clone https://github.com/ChaoningZhang/MobileSAM.git && \
    cd $USER_HOME/MobileSAM/MobileSAMv2 && \
    gdown https://drive.google.com/uc?id=1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE && \
    unzip weight.zip && \
    rm -f weight.zip && \
    sed -i "s/return torch.load(file, map_location='cpu'), file/return torch.load(file, map_location='cpu', weights_only=False), file/" $USER_HOME/MobileSAM/MobileSAMv2/ultralytics/nn/tasks.py

ENV PYTHONPATH="${USER_HOME}/MobileSAM/MobileSAMv2:${PYTHONPATH:-}"

###############################################################################

#### Setup Repo ####

WORKDIR $USER_HOME
RUN git clone https://github.com/maxdoesch/td3_bc.git -b feat-integrating-ftd && \
    cd td3_bc && \
    uv sync --all-extras && \
    uv pip install PyOpenGL_accelerate

###############################################################################