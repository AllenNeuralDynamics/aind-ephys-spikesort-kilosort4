# hash:sha256:c3d1db6a6b43b344dd98bfea0fe0987a2fd8712b9a9d26cfb26d1633e09e7a5d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.30_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2..8.1 \
    aind-log-utils==0.2.3 \
    scipy==1.17.1 \
    spikeinterface[full]==0.104.7
