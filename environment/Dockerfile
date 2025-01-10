# hash:sha256:7a89be51372b02435f84b9534167f94de92441101bcf2fb261a1882a3cdebb82
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.18_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-log-utils==0.2.3 \
    spikeinterface[full]==0.101.2
