FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates gpg-agent kmod

# PYTHON 3.10
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

# ANACONDA
RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
RUN bash /tmp/anaconda.sh -b -p /anaconda \
    && eval "$(/anaconda/bin/conda shell.bash hook)" \
    && conda init \
    && conda update -n base -c defaults conda \
    && conda create python=3.10 --name fsfp \
    && conda activate fsfp \
    && conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia \
    && conda install fastapi
# REPO
RUN mkdir /Pro-FSFP && git clone https://github.com/czeslaw-milosz/Pro-FSFP.git /Pro-FSFP
WORKDIR /Pro-FSFP

# ENVIRONMENT
RUN echo "conda activate fsfp" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# PORTS
EXPOSE 80

# Set entrypoint
COPY entrypoint.sh ./
ENTRYPOINT [ "./entrypoint.sh" ]
# ENTRYPOINT ["bash", "-l"]