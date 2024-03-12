# hash:sha256:afcf783225343f34d1f2eb853e8cf681fe4158d7870e6e2bd2b4ac07e8cdd81e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.19.1 \
    spikeinterface[full]==0.100.1
