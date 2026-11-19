#!/usr/bin/env bash
# 卡组扫盘GA 单轮运行（GitHub Actions 每个 matrix job 调一次）
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SEED="${SEED:-920000}"
ROUND="${ROUND:-0}"
GENS="${GENS:-12}"
LAM="${LAM:-8}"
GAMES="${GAMES:-40}"
VERIFY="${VERIFY:-200}"
python3 "$HERE/scripts/ga_deck_scan.py" \
  --out "$HERE/out_r${ROUND}" \
  --exe "$HERE/bin/cardtest" \
  --data-root "$HERE/data" \
  --rounds 1 --seeds "$SEED" \
  --gens "$GENS" --lam "$LAM" --games "$GAMES" --verify-games "$VERIFY" \
  --threads 2 --rng-seed $((1000 + ROUND)) --objective pt
