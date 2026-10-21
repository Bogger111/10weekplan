FROM registry-1.docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9

ARG MAX_JOBS=2
LABEL org.opencontainers.image.title="Week3 PyTorch3D GPU research environment" \
      org.opencontainers.image.source="https://github.com/facebookresearch/pytorch3d" \
      org.opencontainers.image.revision="33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba" \
      research.torch="2.7.1" \
      research.cuda="12.8" \
      research.cuda.arch="sm_120"

ENV PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=2

COPY requirements.lock source.lock.json pytorch3d-source.tar.gz verify.py /opt/pytorch3d/
COPY wheelhouse/ /opt/pytorch3d/wheelhouse/

RUN python -m pip install --no-index --find-links=/opt/pytorch3d/wheelhouse \
        --no-deps --require-hashes -r /opt/pytorch3d/requirements.lock \
    && python -c "import torch; assert torch.__version__.startswith('2.7.1'); assert torch.version.cuda == '12.8'" \
    && python -c "import hashlib,json,pathlib; p=pathlib.Path('/opt/pytorch3d'); expected=json.loads((p/'source.lock.json').read_text())['source_sha256']; assert hashlib.sha256((p/'pytorch3d-source.tar.gz').read_bytes()).hexdigest() == expected" \
    && mkdir -p /tmp/pytorch3d-src /opt/pytorch3d/built-wheels \
    && tar -xzf /opt/pytorch3d/pytorch3d-source.tar.gz -C /tmp/pytorch3d-src --strip-components=1 \
    && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/usr/local/cuda \
        CUB_HOME=/usr/local/cuda/include MAX_JOBS=${MAX_JOBS} \
        python -m pip wheel --no-deps --no-build-isolation --no-index \
        --wheel-dir=/opt/pytorch3d/built-wheels /tmp/pytorch3d-src \
    && python -m pip install --no-index --no-deps /opt/pytorch3d/built-wheels/pytorch3d-*.whl \
    && python -m pip check \
    && python -m pip freeze --all > /opt/pytorch3d/installed.lock.txt \
    && nvcc --version > /opt/pytorch3d/nvcc-version.txt \
    && g++ --version > /opt/pytorch3d/compiler-version.txt \
    && rm -rf /tmp/pytorch3d-src

RUN python /opt/pytorch3d/verify.py --device cpu --json /opt/pytorch3d/build-verification-cpu.json

WORKDIR /workspace/week3
CMD ["bash"]
