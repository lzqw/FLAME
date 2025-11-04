#!/bin/bash

# 如果任何命令失败，脚本将立即退出
set -e

# --- 脚本开始 ---
echo "=================================================="
echo "开始执行 18 个任务 (环境版本: v5)..."
echo "=================================================="
echo ""

# --- 指令列表 ---


#echo "[Task 1/18] Now running: python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 1 --alg sac"
#python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 20 --alg rf_sac_entient  --noise_scale 0.001 --target_entropy_scale 1.0 --alpha_lr 0.007
#echo "✅ Task 1/18 finished."
#echo ""



python scripts/train_mujoco.py --env Ant-v5 --diffusion_steps 20 --alg rf_sac_ent  --total_step 2000000 --num_vec_envs 10

python scripts/train_mujoco.py --env Ant-v5 --diffusion_steps 20 --alg rf_sac_ent  --total_step 2000000 --target_entropy_scale 1.0 --alpha 0.1 --num_vec_envs 2

python scripts/train_mujoco.py --env Ant-v5 --diffusion_steps 20 --alg rf_sac_ent  --total_step 2000000 --target_entropy_scale 1.0 --alpha 0.05 --num_vec_envs 2

python scripts/train_mujoco.py --env Ant-v5 --diffusion_steps 20 --alg rf_sac_ent  --total_step 2000000 --target_entropy_scale 1.0 --alpha 0.001 --num_vec_envs 2

python scripts/train_mujoco.py --env Ant-v5 --diffusion_steps 20 --alg rf_sac_ent  --total_step 2000000 --target_entropy_scale 1.0 --alpha 0.005 --num_vec_envs 2

# --- 所有任务执行完毕 ---
echo "=================================================="
echo "🎉 所有 18 个任务已全部执行完毕！脚本将退出。"
echo "=================================================="
