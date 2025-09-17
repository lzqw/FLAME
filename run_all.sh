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
python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 1 --alg sac
echo "✅ Task 1/18 finished."
echo ""

echo "[Task 2/18] Now running: python scripts/train_mujoco.py --env Ant-V4 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env Ant-V4 --diffusion_steps 1 --alg sac
echo "✅ Task 2/18 finished."
echo ""

echo "[Task 3/18] Now running: python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 1 --alg sac
echo "✅ Task 3/18 finished."
echo ""

echo "[Task 4/18] Now running: python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 1 --alg sac
echo "✅ Task 4/18 finished."
echo ""

echo "[Task 5/18] Now running: python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 1 --alg sac
echo "✅ Task 5/18 finished."
echo ""

echo "[Task 6/18] Now running: python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 1 --alg sac"
python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 1 --alg sac
echo "✅ Task 6/18 finished."
echo ""

echo "[Task 7/18] Now running: python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 1 --alg rf_sac"
python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 1 --alg rf_sac
echo "✅ Task 7/18 finished."
echo ""

echo "[Task 8/18] Now running: python scripts/train_mujoco.py --env Ant-V4 --diffusion_steps 1 --alg rf_sac"
python scripts/train_mujoco.py --env Ant-V4 --diffusion_steps 1 --alg rf_sac
echo "✅ Task 8/18 finished."
echo ""

echo "[Task 9/18] Now running: python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 1 --alg rf_sac"
python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 1 --alg rf_sac
echo "✅ Task 9/18 finished."
echo ""

echo "[Task 10/18] Now running: python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 1 --alg rf_sac"
python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 1 --alg rf_sac
echo "✅ Task 10/18 finished."
echo ""

echo "[Task 11/18] Now running: python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 1 --alg rf_sac"
python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 1 --alg rf_sac
echo "✅ Task 11/18 finished."
echo ""

echo "[Task 12/18] Now running: python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 1 --alg rf_sac"
python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 1 --alg rf_sac
echo "✅ Task 12/18 finished."
echo ""

echo "[Task 13/18] Now running: python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 20 --alg sdac"
python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 20 --alg sdac
echo "✅ Task 13/18 finished."
echo ""

echo "[Task 14/18] Now running: python scripts/train_mujoco.py --env Ant-V4 --diffusion_steps 20 --alg sdac"
python scripts/train_mujoco.py --env Ant-V4 --diffusion_steps 20 --alg sdac
echo "✅ Task 14/18 finished."
echo ""

echo "[Task 15/18] Now running: python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 20 --alg sdac"
python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 20 --alg sdac
echo "✅ Task 15/18 finished."
echo ""

echo "[Task 16/18] Now running: python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg sdac"
python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg sdac
echo "✅ Task 16/18 finished."
echo ""

echo "[Task 17/18] Now running: python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 20 --alg sdac"
python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 20 --alg sdac
echo "✅ Task 17/18 finished."
echo ""

echo "[Task 18/18] Now running: python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 20 --alg sdac"
python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 20 --alg sdac
echo "✅ Task 18/18 finished."
echo ""

# --- 所有任务执行完毕 ---
echo "=================================================="
echo "🎉 所有 18 个任务已全部执行完毕！脚本将退出。"
echo "=================================================="
