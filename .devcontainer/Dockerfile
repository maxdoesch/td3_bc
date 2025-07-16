FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

###############################################################################

#### Match User and Group id ####

ARG USERNAME=smlr
ARG USER_UID=1000
ARG USER_GID=$USER_UID

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
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
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

SHELL ["/bin/bash", "-c"]

ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh

###############################################################################

#### Python Environment ####

# MobileSAMv2
RUN cd / && \
    git clone https://github.com/ChaoningZhang/MobileSAM.git && \
    cd /MobileSAM/MobileSAMv2 && \
    python -m pip install gdown && \
    gdown https://drive.google.com/uc?id=1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE && \
    unzip weight.zip && \
    rm -f weight.zip && \
    sed -i "s/return torch.load(file, map_location='cpu'), file/return torch.load(file, map_location='cpu', weights_only=False), file/" /MobileSAM/MobileSAMv2/ultralytics/nn/tasks.py && \
    cd .. && python -m pip install -e .

ENV PYTHONPATH="${PYTHONPATH}:/MobileSAM/MobileSAMv2"

###############################################################################

#### Setup Repo ####

RUN cd / && pip install uv && \
    git clone https://github.com/maxdoesch/td3_bc.git && \
    cd /td3_bc && \
    uv sync && \
    pip install -e .

###############################################################################

# Switch to non-root user
USER $USERNAME