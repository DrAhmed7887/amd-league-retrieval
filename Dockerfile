# AMD AI League: one image for Match 2 (OCR) and Match 3 (RAG).
#
# The FINAL stage must start from the mandated base: the harness checks layer identity,
# so never squash or flatten this image.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# pip can silently replace ROCm torch with a CUDA build. Pin whatever torch the base
# ships as a constraint, let pip resolve the rest, then fail the build if torch changed.
COPY requirements.txt /app/requirements.txt
RUN pip list --format=freeze | grep -iE '^(torch|torchvision|torchaudio|triton)==' > /tmp/torch-pins.txt \
 && pip install -c /tmp/torch-pins.txt -r /app/requirements.txt \
 && python3 -c "import torch; v = torch.__version__; assert '+rocm' in v, v; print('torch', v)"

# Weights ship inside the image: the graded run has no network. ~16.3 GiB in 4 shards.
# One layer per shard (each under 5 GiB) so no single layer is huge to push or pull.
ARG MODEL_REPO=Qwen/Qwen3-VL-8B-Instruct
ARG MODEL_REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
ARG MODEL_DIR=/models/Qwen3-VL-8B-Instruct
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='${MODEL_DIR}', allow_patterns=['model-00001-of-00004.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='${MODEL_DIR}', allow_patterns=['model-00002-of-00004.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='${MODEL_DIR}', allow_patterns=['model-00003-of-00004.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='${MODEL_DIR}', allow_patterns=['model-00004-of-00004.safetensors'])"
# Everything else (config, tokenizer, index); shards already present are skipped.
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='${MODEL_DIR}', ignore_patterns=['*.safetensors'])" \
 && python3 -c "import json, pathlib; idx = json.load(open('${MODEL_DIR}/model.safetensors.index.json')); need = set(idx['weight_map'].values()); have = {p.name for p in pathlib.Path('${MODEL_DIR}').glob('*.safetensors')}; assert need <= have, need - have; print('shards ok', sorted(have))"

COPY app.py server.py /app/
COPY engine/ /app/engine/
RUN mkdir -p /app/corpus /app/output /app/index

ENV MC3_READER=qwen \
    MODEL_DIR=/models/Qwen3-VL-8B-Instruct \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Loads the model once and serves app.py over a unix socket. It never exits, which keeps
# the container up for the harness to exec into.
CMD ["python3", "/app/server.py"]
