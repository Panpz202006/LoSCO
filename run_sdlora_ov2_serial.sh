#!/usr/bin/env bash
# ============================================================
# 串行跑 configs/sdlora_ov2/ 下全部 config (52 份)
#   每个跑完才跑下一个; device=5; 日志 -> results_sdlora_ov2/
# 用法: bash run_sdlora_ov2_serial.sh   (需在仓库根目录, 脚本会自动 cd)
# ============================================================
set -u
cd "$(dirname "$0")"

mkdir -p results_sdlora_ov2

for cfg in configs/sdlora_ov2/*.json; do
    name=$(basename "$cfg" .json)
    log="results_sdlora_ov2/sdlora_ov2_${name}.log"
    echo ">>> [$(date '+%F %T')] start : ${cfg}"
    nohup python main.py --device 5 --config "$cfg" > "$log" 2>&1 &
    wait
    echo "<<< [$(date '+%F %T')] done : ${cfg}  (see ${log})"
done

echo "ALL DONE $(date '+%F %T')"
