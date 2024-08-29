# hash:sha256:6874798b1e8043aba7e572a13dd03232e416f34b6639489d4398b4f218cd6abd
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.12_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    spikeinterface[full]==0.101.0
