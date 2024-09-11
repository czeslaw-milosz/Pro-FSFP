FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime AS build

ENV PYTHONUNBUFFERED=1 

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl git ca-certificates gpg-agent unzip

# PYTHON 3.10
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    # python3.10-dev \
    # python3.10-distutils \
    # python3.10-lib2to3 \
    # python3.10-gdbm \
    pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

# ANACONDA
COPY requirements.txt /tmp/requirements.txt
RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash /tmp/anaconda.sh -b -p /anaconda \
    && eval "$(/anaconda/bin/conda shell.bash hook)" \
    && conda init \
    && conda update -n base -c defaults conda \
    && conda create python=3.10 --name fsfp \
    && conda activate fsfp \
    && conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && conda install -c conda-forge conda-pack

# PACKAGE ENVIRONMENT WITH CONDA-PACK FOR MULTISTAGE BUILD
RUN conda-pack -n example -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
RUN /venv/bin/conda-unpack

# RUNTIME STAGE
FROM debian:stable-slim AS runtime
COPY --from=build /venv /venv

# MODEL CHECKPOINTS
ADD assets /root/assets
RUN unzip /root/assets/checkpoints.zip -d /root \
    && mv /root/assets/huggingface/ /root/.cache/huggingface/ \ 
    && rm -r /root/assets
ENV HF_HOME=/root/.cache/huggingface

# REPO
ADD "https://api.github.com/repos/czeslaw-milosz/Pro-FSFP/commits?per_page=1" latest_commit
RUN mkdir /Pro-FSFP && git clone https://github.com/czeslaw-milosz/Pro-FSFP.git /Pro-FSFP \ 
    && rm -r /Pro-FSFP/checkpoints \
    && mv /root/checkpoints /Pro-FSFP
WORKDIR /Pro-FSFP

# ENVIRONMENT
# RUN echo "conda activate fsfp" >> ~/.bashrc

# PORTS
EXPOSE 8080

# Set entrypoint
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]