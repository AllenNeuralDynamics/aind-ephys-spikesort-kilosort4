# hash:sha256:92991a9228b81e3a27b1adab24ec3c4e82990f3b6a6d32be6ddf2fcd8dbf14cf
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.30_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    aind-log-utils==0.2.3 \
    scipy==1.17.1 \
    spikeinterface[full]==0.104.0
