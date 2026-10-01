# hash:sha256:6751501bb0717a0a17d517ae2a9128832cdd86f9a9cc99afbf2efe963dbd870f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.1.7_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    log-schema==0.2.10.dev1 \
    scipy==1.17.1 \
    spikeinterface[full]==0.105.0
