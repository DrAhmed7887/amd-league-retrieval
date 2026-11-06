#!/usr/bin/env bash
# Build day, on a Linux x86-64 machine with Docker and an AMD GPU (AMD Developer Cloud).
# Needs ~120 GB free disk. Does NOT push: log in and push by hand after every check passes.
#
#   bash scripts/build_and_check.sh amd-league:v1
set -euo pipefail
TAG="${1:-amd-league:local}"
NAME="amdl-check-$$"
cd "$(dirname "$0")/.."

if [ ! -d kit ]; then
  curl -sSLo /tmp/mc3-starter-kit.zip https://storage.googleapis.com/lablab-static-eu/share/mc3-starter-kit.zip
  python3 -m zipfile -e /tmp/mc3-starter-kit.zip /tmp/ && mv /tmp/mc3-starter-kit kit && sh kit/setup.sh
fi

echo "== 1. build"
docker build -t "$TAG" .

echo "== 2. torch must still be the ROCm build"
docker run --rm "$TAG" python3 -c "import torch; v = torch.__version__; print(v); assert '+rocm' in v"

echo "== 3. uncompressed size (limit 60 GiB)"
docker history --no-trunc --format '{{.Size}}' "$TAG" | python3 -c "
import sys, re
units = {'B': 1, 'kB': 1e3, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9, 'TB': 1e12}
total = 0
for line in sys.stdin:
    m = re.match(r'([\d.]+)\s*([kKMGT]?B)', line.strip())
    if m: total += float(m.group(1)) * units[m.group(2)]
gib = total / 2**30
print(f'{gib:.1f} GiB'); sys.exit(0 if gib < 59 else 1)"

echo "== 4. real GPU run: no network, DAC_OVERRIDE dropped, sample questions + two OCR images"
docker run -d --name "$NAME" --network none --cap-drop DAC_OVERRIDE \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host "$TAG" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 120); do docker logs "$NAME" 2>&1 | grep -q "model loaded\|model service failed" && break; sleep 5; done
docker logs "$NAME" 2>&1 | tail -3
docker exec "$NAME" mkdir -p /app/corpus /app/output
docker cp kit/mc3-corpus/. "$NAME":/app/corpus
docker exec "$NAME" chmod 000 /app/corpus/vendor/internal_audit.txt
python3 scripts/score_container.py "$NAME"
for img in specs/backplane_pinout.png support/asset_label.jpg; do
  docker exec "$NAME" python3 /app/app.py --input-image "/app/corpus/$img"
  docker exec "$NAME" cat "/app/output/$(basename "${img%.*}")_output.json"; echo
done
echo "peak VRAM while running:"; docker exec "$NAME" sh -c "amd-smi metric --mem 2>/dev/null | grep -i used || rocm-smi --showmeminfo vram" || true

echo "== 5. AMD's own self-check (it runs without a GPU, so answers will be empty there; it checks the contract)"
python3 kit/selfcheck.py "$TAG" kit/mc3-corpus
