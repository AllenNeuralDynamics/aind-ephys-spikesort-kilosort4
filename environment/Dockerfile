# hash:sha256:37007c84699b04ee6ec48c2c08057e34413320c9ea8c66fc47b9bfaca3d606d2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.30_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    aind-log-utils==0.2.3 \
    scipy==1.17.1 \
    spikeinterface[full]==0.103.2
