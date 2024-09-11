FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime AS build

ENV PYTHONUNBUFFERED=1 

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils wget

# ANACONDA; PACKAGE ENVIRONMENT WITH CONDA-PACK FOR MULTISTAGE BUILD
COPY environment.yml /tmp/environment.yml
RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash /tmp/miniconda.sh -b -p /miniconda \
    && eval "$(/miniconda/bin/conda shell.bash hook)" \
    && conda init \
    && conda update -n base -c defaults conda \
    && conda env create -f /tmp/environment.yml \
    && conda install -c conda-forge conda-pack \
    && conda clean --all \
    && conda-pack -n fsfp -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
RUN /venv/bin/conda-unpack

# RUNTIME STAGE
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS runtime

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    gcc software-properties-common \
    build-essential apt-utils \
    wget curl git ca-certificates gpg-agent unzip
COPY --from=build /venv /venv

# ACTIVATE ENVIRONMENT + MANUALLY INSTALL LEARN2LEARN
SHELL ["/bin/bash", "-c"] 
RUN source /venv/bin/activate \
    && pip install --upgrade pip \
    && pip install learn2learn>=0.2.0

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

# PORTS
EXPOSE 8080

# SET ENTRYPOINT
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]