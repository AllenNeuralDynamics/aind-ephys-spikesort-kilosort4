# hash:sha256:c94f893836d4d0ee35b9c51b20ae853fc00d430e7d94088defce145a1c602a60
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.0.30_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    aind-log-utils==0.2.3 \
    scipy==1.17.1 \
    spikeinterface[full]==0.103.0
