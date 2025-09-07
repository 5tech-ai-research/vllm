ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
ARG FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

ARG PYTORCH_CUDA_INDEX_BASE_URL=https://download.pytorch.org/whl
ARG PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL=https://download.pytorch.org/whl/nightly

ARG PIP_KEYRING_PROVIDER=disabled
ARG UV_KEYRING_PROVIDER=${PIP_KEYRING_PROVIDER}

ARG INSTALL_KV_CONNECTORS=false

#################### BASE BUILD IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG TARGETPLATFORM
ARG INSTALL_KV_CONNECTORS=false
ENV DEBIAN_FRONTEND=noninteractive

ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && if [ ! -z ${DEADSNAKES_MIRROR_URL} ] ; then \
        if [ ! -z "${DEADSNAKES_GPGKEY_URL}" ] ; then \
            mkdir -p -m 0755 /etc/apt/keyrings ; \
            curl -L ${DEADSNAKES_GPGKEY_URL} | gpg --dearmor > /etc/apt/keyrings/deadsnakes.gpg ; \
            sudo chmod 644 /etc/apt/keyrings/deadsnakes.gpg ; \
            echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] ${DEADSNAKES_MIRROR_URL} $(lsb_release -cs) main" > /etc/apt/sources.list.d/deadsnakes.list ; \
        fi ; \
    else \
        for i in 1 2 3; do \
            add-apt-repository -y ppa:deadsnakes/ppa && break || \
            { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
        done ; \
    fi \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

RUN apt-get install -y gcc-10 g++-10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10
RUN gcc --version

RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

COPY requirements/common.txt requirements/common.txt
COPY requirements/cuda.txt requirements/cuda.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/cuda.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

ARG torch_cuda_arch_list='7.0 7.5 8.0 8.9 9.0 10.0 12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

#################### DEPENDENCIES BUILD STAGE ####################
FROM base AS dependencies-build
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

# Install build dependencies
COPY requirements/build.txt requirements/build.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# Copy only files needed for building extensions (not Python code)
COPY setup.py setup.py
COPY pyproject.toml pyproject.toml
COPY CMakeLists.txt CMakeLists.txt
COPY cmake cmake
COPY csrc csrc
COPY vllm/__init__.py vllm/__init__.py
COPY vllm/_version.py vllm/_version.py
COPY vllm/compilation vllm/compilation
COPY vllm/_custom_ops.py vllm/_custom_ops.py
COPY vllm/utils.py vllm/utils.py
COPY vllm/logger.py vllm/logger.py
COPY vllm/platforms vllm/platforms
COPY vllm/config.py vllm/config.py
COPY vllm/envs.py vllm/envs.py

ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

ARG USE_SCCACHE
ARG SCCACHE_DOWNLOAD_URL=https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz
ARG SCCACHE_ENDPOINT
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0

ARG VLLM_USE_PRECOMPILED=""
ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}
ENV CCACHE_DIR=/root/.cache/ccache

# Build CUDA extensions only
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && curl -L -o sccache.tar.gz ${SCCACHE_DOWNLOAD_URL} \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl \
        && if [ ! -z ${SCCACHE_ENDPOINT} ] ; then export SCCACHE_ENDPOINT=${SCCACHE_ENDPOINT} ; fi \
        && export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} \
        && export SCCACHE_REGION=${SCCACHE_REGION_NAME} \
        && export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \
        && export SCCACHE_IDLE_TIMEOUT=0 \
        && export CMAKE_BUILD_TYPE=Release \
        && export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" \
        && export VLLM_DOCKER_BUILD_CONTEXT=1 \
        && sccache --show-stats \
        && python3 setup.py build_ext --inplace \
        && sccache --show-stats; \
    else \
        rm -rf .deps && \
        mkdir -p .deps && \
        export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" && \
        export VLLM_DOCKER_BUILD_CONTEXT=1 && \
        python3 setup.py build_ext --inplace; \
    fi

#################### FINAL BUILD STAGE ####################
FROM dependencies-build AS build

# Now copy all the Python source code
COPY vllm vllm
COPY tests tests
COPY examples examples
COPY benchmarks benchmarks
COPY tools tools

# Build the wheel (fast since extensions are already built)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" && \
    export VLLM_DOCKER_BUILD_CONTEXT=1 && \
    python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

# Check wheel size
COPY .buildkite/check-wheel-size.py check-wheel-size.py
ARG VLLM_MAX_SIZE_MB=400
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=true
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check."; \
    fi

#################### DEV IMAGE ####################
FROM dependencies-build AS dev

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y libnuma-dev && rm -rf /var/lib/apt/lists/*

COPY requirements/lint.txt requirements/lint.txt
COPY requirements/test.txt requirements/test.txt
COPY requirements/dev.txt requirements/dev.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/dev.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# Copy Python source code for development
COPY vllm vllm
COPY tests tests
COPY examples examples
COPY benchmarks benchmarks

#################### vLLM INSTALLATION IMAGE ####################
FROM ${FINAL_BASE_IMAGE} AS vllm-base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG INSTALL_KV_CONNECTORS=false
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM

SHELL ["/bin/bash", "-c"]

ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl wget sudo vim python3-pip \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 \
    && if [ ! -z ${DEADSNAKES_MIRROR_URL} ] ; then \
        if [ ! -z "${DEADSNAKES_GPGKEY_URL}" ] ; then \
            mkdir -p -m 0755 /etc/apt/keyrings ; \
            curl -L ${DEADSNAKES_GPGKEY_URL} | gpg --dearmor > /etc/apt/keyrings/deadsnakes.gpg ; \
            sudo chmod 644 /etc/apt/keyrings/deadsnakes.gpg ; \
            echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] ${DEADSNAKES_MIRROR_URL} $(lsb_release -cs) main" > /etc/apt/sources.list.d/deadsnakes.list ; \
        fi ; \
    else \
        for i in 1 2 3; do \
            add-apt-repository -y ppa:deadsnakes/ppa && break || \
            { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
        done ; \
    fi \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv libibverbs-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        uv pip install --system \
            --index-url ${PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            "torch==2.8.0.dev20250318+cu128" "torchvision==0.22.0.dev20250319" ; \
        uv pip install --system \
            --index-url ${PYTORCH_CUDA_NIGHTLY_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            --pre pytorch_triton==3.3.0+gitab727c40 ; \
    fi

# Install vllm wheel
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# Install FlashInfer
ARG FLASHINFER_GIT_REPO="https://github.com/flashinfer-ai/flashinfer.git"
ARG FLASHINFER_GIT_REF="v0.2.14.post1"
ARG FLASHINFER_AOT_COMPILE=false
RUN --mount=type=cache,target=/root/.cache/uv bash - <<'BASH'
  . /etc/environment
    git clone --depth 1 --recursive --shallow-submodules \
        --branch ${FLASHINFER_GIT_REF} \
        ${FLASHINFER_GIT_REPO} flashinfer
    pushd flashinfer
        if [ "${FLASHINFER_AOT_COMPILE}" = "true" ]; then
            if [[ "${CUDA_VERSION}" == 11.* ]]; then
                FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9"
            elif [[ "${CUDA_VERSION}" == 12.[0-7]* ]]; then
                FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a"
            else
                FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 12.0"
            fi
            echo "🏗️  Installing FlashInfer with AOT compilation for arches: ${FI_TORCH_CUDA_ARCH_LIST}"
            TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
                python3 -m flashinfer.aot
            TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
                uv pip install --system --no-build-isolation . \
                --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
            TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
                python3 -m flashinfer --download-cubin || echo "WARNING: Failed to download flashinfer cubins."
        else
            echo "🏗️  Installing FlashInfer without AOT compilation in JIT mode"
            uv pip install --system . \
                --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
        fi
    popd
    rm -rf flashinfer
BASH

COPY examples examples
COPY benchmarks benchmarks
COPY ./vllm/collect_env.py .

RUN --mount=type=cache,target=/root/.cache/uv \
. /etc/environment && \
uv pip list

COPY requirements/build.txt requirements/build.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# Install DeepGEMM
ARG DEEPGEMM_GIT_REF
COPY tools/install_deepgemm.sh /tmp/install_deepgemm.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    VLLM_DOCKER_BUILD_CONTEXT=1 /tmp/install_deepgemm.sh --cuda-version "${CUDA_VERSION}" ${DEEPGEMM_GIT_REF:+--ref "$DEEPGEMM_GIT_REF"}

# Install EP kernels and NixL
COPY tools/ep_kernels/install_python_libraries.sh install_python_libraries.sh
COPY tools/install_nixl.sh install_nixl.sh
ENV CUDA_HOME=/usr/local/cuda
RUN export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a+PTX}" \
    && bash install_python_libraries.sh \
    && bash install_nixl.sh --force

#################### TEST IMAGE ####################
FROM vllm-base AS test

ADD . /vllm-workspace/

ARG PYTHON_VERSION
ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    if [ "$CUDA_MAJOR" -ge 12 ]; then \
        uv pip install --system -r requirements/dev.txt; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e tests/vllm_test_utils

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER 1

COPY vllm/v1 /usr/local/lib/python${PYTHON_VERSION}/dist-packages/vllm/v1

RUN mkdir src
RUN mv vllm src/vllm

#################### OPENAI API SERVER ####################
FROM ${FINAL_BASE_IMAGE} AS vllm-openai-base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG INSTALL_KV_CONNECTORS=false
WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 -m pip install uv

ARG INSTALL_VLLM_WHEEL=false
COPY --from=build /workspace/dist /tmp/dist
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_VLLM_WHEEL" = "true" ]; then \
        uv pip install --system /tmp/dist/*.whl; \
    fi

COPY requirements/kv_connectors.txt requirements/kv_connectors.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then \
        uv pip install --system -r requirements/kv_connectors.txt; \
    fi; \
    uv pip install --system accelerate hf_transfer boto3 'timm==0.9.10'

ENV VLLM_USAGE_SOURCE production-docker-image

FROM vllm-openai-base AS vllm-openai
COPY vllm vllm
COPY examples examples
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]