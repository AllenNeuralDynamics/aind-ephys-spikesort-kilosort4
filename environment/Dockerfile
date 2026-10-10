# hash:sha256:50f5dc8af7c11ce3da1d34e99952838a2d41c38b045913f1c34e7e5ea19883be
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.1.7_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    log-schema==0.2.10.dev1 \
    scipy==1.17.1 \
    spikeinterface[full]==0.104.7
