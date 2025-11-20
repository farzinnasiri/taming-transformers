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

RUN apt-get update && apt-get install -y \
    build-essential ca-certificates git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Add user with its groups
USER 0
RUN IFS=";" read -a myarr <<< "$USER_GADD_ARGS" && \
    unset 'myarr[${#arr[@]}-1]' && \
    for i in ${!myarr[@]}; do \
        if [[ ${myarr[$i]} != "sudo"* ]]; then \
            groupadd -f -g ${myarr[$i]}; \
        fi \
    done
RUN  useradd -u $USER_UID -g $USER_GID -m $USERNAME
RUN for g in $USER_GNAMES; do \
        usermod -a -G $g $USERNAME; \
    done
# Allow sudo
RUN     apt-get install -y sudo \
        && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
        && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# -- Create folders under root for following services
USER 0

# Enable jupyter
RUN mkdir -p /.local
RUN chmod -R 777 /.local

# Enable avalanche
RUN mkdir -p /.avalanche
RUN chmod -R 777 /.avalanche

# Enable wandb
RUN mkdir -p /.config
RUN chmod -R 777 /.config
RUN touch /.netrc
RUN chmod 777 /.netrc
RUN mkdir -p /.cache
RUN chmod -R 777 /.cache

USER $USERNAME

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      tensorflow-cpu==2.9.0 \
      numpy==1.23.5 \
      scipy==1.10.1 \
      tqdm \
      requests


COPY evaluator.py /work/evaluator.py

CMD ["bash"]

# Build:
#   docker build -t nasiri/evaluator:latest -f eval/Dockerfile .
# Run:
#   docker run --rm -it -v /srv/npz:/data -w /data nasiri/evaluator:latest \
#     python /work/evaluator.py /data/VIRTUAL_imagenet256_labeled.npz /data/1733664935_imagenet256_recon.npz