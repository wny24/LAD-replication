# Latent Diffusion Policy（双臂）

在已经训好的 RobotCLIP encoder / decoder 上，按 Chi et al., *Diffusion Policy* (IJRR 2024) 训 **双臂** 视觉运动策略。

对照 LAD 思路：

- CLIP 的 \(q_i, p_i\) **全程冻结**
- 左右手部动作**各自**过同一冻结 encoder，得到 \(z_L, z_R\)（各 32 维）
- 扩散目标：`[z_L, z_R, q_arm_L, q_arm_R]`（默认 64+14=78 维）
- 推理时用 \(p_j\) **分别**解左右手，一次输出完整双臂指令

网络是 Chi 的 1D temporal U-Net（FiLM 条件、squared-cosine 噪声、\(\epsilon\)-prediction）。视觉 backbone 为 ResNet18。

> **注意**：此前单臂拆分训练的 `checkpoints/policy_*.pth` 与当前 `n_arms=2` **不兼容**，需用双臂数据重新训 diffusion（CLIP 不用重训）。

## 安装

```bash
conda activate LAD
pip install hydra-core omegaconf torchvision tqdm opencv-python-headless
cd /home/wny24/LAD-replication/robot_clip
pip install -e .
```

## 真机 transport → 双臂 episode

原始 zip 已解压在：

- `data/raw_xhand/transport-test/`：每侧 `(T,19)=7` 臂 + `12` 指
- `data/raw_g2/transport_g2_controller/`：每侧 `(T,8)=7` 臂 + `1` 夹爪（`0..1000`）

转换（**一条 demo → 一个双臂 npz**，会删除旧的 `*_left.npz` / `*_right.npz`）：

```bash
cd /home/wny24/LAD-replication/robot_clip/diffusion
python scripts/convert_transport.py --overwrite
# 试跑：python scripts/convert_transport.py --max-episodes 2 --overwrite
```

写出 `data/{xhand,g2}/episode_N.npz`：

| 输出 key | xhand | g2 | 用途 |
|---|---|---|---|
| `action` | `(T, 24)=[L12,R12]` | `(T, 2)=[L1,R1]` 开合÷1000→`[0,1]` | 左右各进冻结 CLIP → \(z_L\|z_R\) |
| `wrist_pose` | `(T, 14)=[L7,R7]` 臂指令 | 同左 | **不进 CLIP**，与 latent 拼接扩散 |
| `lowdim_obs` | `(T, 14)=[L7,R7]` 臂 `qpos` | 同左 | 观测条件 |
| `image` | `(T, 84, 84, 3)` | 同左 | 共享相机 |

`config/train.yaml` 默认：`n_arms=2`，`wrist_dim=14`，`obs.lowdim_dim=14`。

## 训练

```bash
conda activate LAD
cd /home/wny24/LAD-replication/robot_clip/diffusion

python train.py \
  clip.checkpoint=/home/wny24/LAD-replication/robot_clip/checkpoints/model_epoch_250.pth \
  training.device=cuda \
  training.gpu_id=0
```

常用覆盖：

```bash
python train.py n_arms=2 wrist_dim=14 obs.lowdim_dim=14 training.batch_size=128
```

U-Net 输入维 = `n_arms * 32 + wrist_dim` = **78**（默认）。checkpoint 仍写在 `checkpoints/policy_step_*.pth`；**不含** CLIP 权重。

## 推理（双臂一次生成）

```bash
cd /home/wny24/LAD-replication/robot_clip/diffusion

python infer.py \
  --checkpoint checkpoints/policy_latest.pth \
  --episode data/xhand/episode_0.npz \
  --embodiment xhand \
  --t 0 \
  --ddim-steps 16 \
  --out checkpoints/sample_xhand.npz

python infer.py \
  --checkpoint checkpoints/policy_latest.pth \
  --episode data/g2/episode_0.npz \
  --embodiment g2 \
  --t 0 \
  --ddim-steps 16 \
  --out checkpoints/sample_g2.npz
```

`sample_*.npz` 中：

- `action`：`(H, 24)` 或 `(H, 2)`，已是 **左右手拼在一起**
- `wrist_pose`：`(H, 14)` 左右臂指令
- `latent`：`(H, 64)` = \(z_L\|z_R\)

## 训练数据流（双臂）

1. 窗口：观测 `n_obs_steps` 帧 + 未来 `horizon` 步双臂动作。
2. 冻结 \(q_i\)：左手 / 右手分别编码 → \(z_L, z_R\)（各 32）。
3. 拼 `[z_L, z_R, q_arm_L, q_arm_R]`，再做 diffusion 侧 Gaussian 归一化。
4. U-Net 预测噪声；只更新视觉编码器与 U-Net。
5. 推理：DDIM → 拆 latent / wrist → \(p_j(z_L), p_j(z_R)\) → 拼接手部动作。

## 重要配置

| 项 | 默认 | 含义 |
|---|---|---|
| `n_arms` | `2` | 臂数；左右手各过一次 CLIP。 |
| `wrist_dim` | `14` | 左右臂关节指令（7+7）。 |
| `obs.lowdim_dim` | `14` | 左右臂实测 `qpos`。 |
| `horizon` | `16` | 预测序列长度。 |
| `clip.checkpoint` | `../checkpoints/model_epoch_250.pth` | 冻结 RobotCLIP。 |
| `training.steps` | `90000` | 优化步数。 |
| `training.batch_size` | `256` | batch size。 |
| `training.ema_decay` | `0.999` | 推理用 EMA。 |
