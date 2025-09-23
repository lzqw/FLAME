#!/bin/bash

# 如果任何命令失败，脚本将立即退出
set -e

# --- 脚本开始 ---
echo "=================================================="
echo "开始执行 18 个任务 (环境版本: v5)..."
echo "=================================================="
echo ""

# --- 指令列表 ---


echo "[Task 1/18] Now running: python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg rf_sac_estient  --noise_scale 0.1 --target_entropy_scale 1.0
echo "✅ Task 1/18 finished."
echo ""

echo "[Task 1/18] Now running: python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg rf_sac_estient  --noise_scale 0.1 --target_entropy_scale 2.0
echo "✅ Task 1/18 finished."
echo ""

echo "[Task 1/18] Now running: python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg rf_sac_estient  --noise_scale 0.1 --target_entropy_scale 0.5
echo "✅ Task 1/18 finished."
echo ""

echo "[Task 1/18] Now running: python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg rf_sac_estient  --noise_scale 0.2 --target_entropy_scale 0.5
echo "✅ Task 1/18 finished."
echo ""

echo "[Task 1/18] Now running: python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg rf_sac_estient  --noise_scale 0.05 --target_entropy_scale 0.1
echo "✅ Task 1/18 finished."
echo ""
# --- 所有任务执行完毕 ---
echo "=================================================="
echo "🎉 所有 18 个任务已全部执行完毕！脚本将退出。"
echo "=================================================="
