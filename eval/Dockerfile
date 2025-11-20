FROM python:3.8-slim

ARG USERNAME
ARG USER_UID
ARG USER_GID=$USER_UID
ARG USER_GNAMES
ARG USER_GADD_ARGS
ARG PIP_REQ_FILE
ARG APT_REQ_FILE

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get update && apt-get install -y \
    build-essential ca-certificates git wget curl \
    sudo passwd && rm -rf /var/lib/apt/lists/*

# Create folders under root for following services
RUN mkdir -p /.local /.config /.cache /.avalanche && chmod -R 777 /.local /.config /.cache /.avalanche && \
    touch /.netrc && chmod 777 /.netrc

# Add user with its groups
RUN IFS=";" read -a myarr <<< "$USER_GADD_ARGS" && \
    unset 'myarr[${#arr[@]}-1]' && \
    for i in ${!myarr[@]}; do \
        if [[ ${myarr[$i]} != "sudo"* ]]; then \
            groupadd -f -g ${myarr[$i]}; \
        fi \
    done
RUN useradd -u $USER_UID -g $USER_GID -m $USERNAME && \
    for g in $USER_GNAMES; do usermod -a -G $g $USERNAME; done && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

USER root
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      tensorflow-cpu==2.9.0 \
      numpy==1.23.5 \
      scipy==1.10.1 \
      tqdm \
      requests

# Optional apt/pip requirements via files
ADD $APT_REQ_FILE /apt_requirements.txt
RUN apt-get update && cat /apt_requirements.txt | xargs apt-get install -y

USER $USERNAME
WORKDIR /home/$USERNAME/exp

COPY evaluator.py /work/evaluator.py

CMD ["bash"]

# Build:
#   docker build -t nasiri/evaluator:latest -f eval/Dockerfile .
# Run:
#   docker run --rm -it -v /srv/npz:/data -w /data nasiri/evaluator:latest \
#     python /work/evaluator.py /data/VIRTUAL_imagenet256_labeled.npz /data/1733664935_imagenet256_recon.npz