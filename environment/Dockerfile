<<<<<<< HEAD
# hash:sha256:4f6c3fe9f051643d1780ca41ba87ff7431414b3731e6a063a6abadf935ec6186
=======
# hash:sha256:0e6716d19a37c3bee117fa9cdd129023bfe45d364423b9d931713b46a732fe17
>>>>>>> origin/main
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/kilosort4-base:4.1.7_cuda-12.0.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    aind-log-utils==0.2.3 \
    scipy==1.17.1 \
    spikeinterface[full]==0.104.7
