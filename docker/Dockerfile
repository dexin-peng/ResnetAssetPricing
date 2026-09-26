ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04
ARG UV_VERSION=0.7.19
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM ${CUDA_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.13.5

ENV VIRTUAL_ENV=/opt/workspace-python \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    PATH="/opt/workspace-python/bin:/usr/local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY --from=uv /uv /uvx /usr/local/bin/

RUN test "$(uname -m)" = x86_64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgomp1 libstdc++6 tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ /opt/resassetpricing-build/requirements/
COPY install-python-stack/ /opt/resassetpricing-build/install-python-stack/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# Keep the large PyTorch download separate so a later failure can reuse its layer.
RUN --mount=type=cache,target=/opt/uv-cache \
    bash /opt/resassetpricing-build/install-python-stack/00-bootstrap.sh "$PYTHON_VERSION"
RUN --mount=type=cache,target=/opt/uv-cache \
    bash /opt/resassetpricing-build/install-python-stack/10-common-packages.sh
RUN --mount=type=cache,target=/opt/uv-cache \
    bash /opt/resassetpricing-build/install-python-stack/20-data-packages.sh
RUN --mount=type=cache,target=/opt/uv-cache \
    bash /opt/resassetpricing-build/install-python-stack/30-pytorch-wheels.sh
RUN bash /opt/resassetpricing-build/install-python-stack/90-finalize.sh \
    && chmod 0755 /usr/local/bin/entrypoint.sh \
    && rm -rf /opt/resassetpricing-build

WORKDIR /workspace
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
