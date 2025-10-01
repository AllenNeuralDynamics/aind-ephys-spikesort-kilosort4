# hash:sha256:db4f51292cef349f04b53145ef4dc033ec9ab7c4eb5ad6caed10af323fd004b0

ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.30_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema \
    aind-log-utils==0.2.3 \
    spikeinterface[full]==0.103.0
