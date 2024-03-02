# hash:sha256:7647af921e0faaecca91bf3971216ed4d7e1765577da32597fef555b74fac52c
FROM registry.codeocean.allenneuraldynamics.org/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.19.1

COPY postInstall /
RUN /postInstall
