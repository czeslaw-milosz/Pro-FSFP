FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

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
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

# ANACONDA
COPY requirements.txt /tmp/requirements.txt
RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
RUN bash /tmp/anaconda.sh -b -p /anaconda \
    && eval "$(/anaconda/bin/conda shell.bash hook)" \
    && conda init \
    && conda update -n base -c defaults conda \
    && conda create python=3.10 --name fsfp \
    && conda activate fsfp \
    && conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# MODEL CHECKPOINTS
COPY assets/ /root/
RUN unzip /root/assets/checkpoints.zip -d /root  \ 
    && mv assets/huggingface /root/.cache/huggingface \ 
    && rm -r /root/assets
# COPY huggingface_cache/huggingface /root/.cache/huggingface

# REPO
ADD "https://api.github.com/repos/czeslaw-milosz/Pro-FSFP/commits?per_page=1" latest_commit
RUN mkdir /Pro-FSFP && git clone https://github.com/czeslaw-milosz/Pro-FSFP.git /Pro-FSFP \ 
    && rm -r /Pro-FSFP/checkpoints \
    && mv /root/checkpoints /Pro-FSFP
WORKDIR /Pro-FSFP
ENV HF_HOME=/root/.cache/huggingface

# ENVIRONMENT
RUN echo "conda activate fsfp" >> ~/.bashrc

# PORTS
EXPOSE 8080

# Set entrypoint
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]