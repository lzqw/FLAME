#!/bin/bash

# 如果任何命令失败，脚本将立即退出
set -e


# --- 指令列表 ---
CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Ant-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Reacher-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Pusher-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Humanoid-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 2000000 --diffusion_steps_test 20 --seed 1

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env InvertedDoublePendulum-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 1


CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 2

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Ant-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 2

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 2

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 2

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 2

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 2

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Reacher-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 2

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Pusher-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 2

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env Humanoid-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 2000000 --diffusion_steps_test 20 --seed 2

CUDA_VISIBLE_DEVICES=0 python scripts/train_mujoco.py --env InvertedDoublePendulum-v5 --diffusion_steps 20 --alg dipo  --noise_scale 0.01 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 20 --seed 2

