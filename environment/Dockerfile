# hash:sha256:5c95d99e29c6fbdbd99e8fc7b87d7efa1c94f20e928fba48a5363655f0baabfd
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.16_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    spikeinterface[full]==0.101.1
