#!/usr/bin/env bash
# round2: 围绕最优点(lo=0.001,ls=0.0001,mu=0.001)的精细消融, 串行 device=5
set -u
cd "$(dirname "$0")"
mkdir -p results_sdlora_ov2
echo ">>> start: imagenet-a_seed=1993_lo=0.00003_ls_0.0001_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.00003_ls_0.0001_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.00003_ls_0.0001_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.00003_ls_0.0001_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0001_ls_0.0001_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0001_ls_0.0001_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0001_ls_0.0001_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0001_ls_0.0001_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.0001_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.0001_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.0001_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.0001_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.0001_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.0001_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.0001_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.0001_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.006_ls_0.0001_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.006_ls_0.0001_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.006_ls_0.0001_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.006_ls_0.0001_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.01_ls_0.0001_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.01_ls_0.0001_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.01_ls_0.0001_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.01_ls_0.0001_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.00003_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.00003_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.00003_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.00003_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.0003_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.0003_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.0003_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.0003_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.001_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.001_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.001_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.001_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.003_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.003_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.003_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.003_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.01_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.01_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.01_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.01_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.0001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.0001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.0001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.0001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.0003.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.0003.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.0003.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.0003.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.003.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.003.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.003.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.003.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.01.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.01.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.01.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.01.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.03.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.03.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.03.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.03.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.1.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.1.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.1.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.0001_mu=0.1.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.00003_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.00003_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.00003_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.00003_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.00003_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.00003_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.00003_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.00003_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.0001_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.0001_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.0001_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.0001_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.0003_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.0003_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.0003_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.0003_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.0003_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.0003_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.0003_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.0003_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.001_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.001_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.001_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.001_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.001_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.001_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.001_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.001_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.00003_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.00003_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.00003_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.00003_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.0003_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.0003_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.0003_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.0003_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.001_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.001_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.001_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.001_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.00003_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.00003_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.00003_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.00003_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.00003_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.00003_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.00003_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.00003_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.0001_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.0001_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.0001_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.0001_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.0003_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.0003_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.0003_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.0003_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.0003_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.0003_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.0003_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.0003_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.001_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.001_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.001_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.001_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.001_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.001_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.001_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.001_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.003_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.003_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.003_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.003_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.01_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.01_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.01_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.01_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.03_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.03_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.03_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.03_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.001_ls_0.03_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.001_ls_0.03_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.001_ls_0.03_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.001_ls_0.03_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.003_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.003_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.003_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.003_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.003_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.003_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.003_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.003_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.01_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.01_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.01_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.01_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.01_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.01_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.01_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.01_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.03_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.03_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.03_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.03_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.0003_ls_0.03_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.0003_ls_0.03_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.0003_ls_0.03_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.0003_ls_0.03_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.003_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.003_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.003_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.003_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.003_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.003_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.003_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.003_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.01_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.01_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.01_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.01_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.01_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.01_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.01_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.01_mu=0.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.03_mu=0.001.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.03_mu=0.001.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.03_mu=0.001.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.03_mu=0.001.json"
echo ">>> start: imagenet-a_seed=1993_lo=0.003_ls_0.03_mu=0.json"
nohup python main.py --device 5 --config "configs/sdlora_ov2/imagenet-a_seed=1993_lo=0.003_ls_0.03_mu=0.json" > "results_sdlora_ov2/sdlora_ov2_imagenet-a_seed=1993_lo=0.003_ls_0.03_mu=0.log" 2>&1 &
wait
echo "<<< done : imagenet-a_seed=1993_lo=0.003_ls_0.03_mu=0.json"
echo "ROUND2 ALL DONE"
