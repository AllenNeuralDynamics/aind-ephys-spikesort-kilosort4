# hash:sha256:b9b9979003b48fbb430be14b9d57a9c163f457698fb2b674b3e67274d027a362
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.1.7_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    aind-log-utils==0.2.8 \
    dartsort==0.5.30 \
    scipy==1.17.1 \
    spikeinterface[full]==0.105.0
