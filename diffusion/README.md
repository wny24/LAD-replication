# Latent Diffusion Policy

在已经训好的 RobotCLIP encoder / decoder 上，按 Chi et al., *Diffusion Policy* (IJRR 2024) 在 **latent action + wrist pose** 上训视觉运动策略。

对照 LAD 论文附录：

- CLIP 的 \(q_i, p_i\) **全程冻结**
- 扩散目标是共享 latent 末端动作（32 维）拼上 **不进 CLIP** 的腕部位姿
- 推理时用目标本体的 decoder \(p_j\) 把 denoised latent 解成关节 / 夹爪开合

网络是 Chi 的 1D temporal U-Net（FiLM 条件、squared-cosine 噪声、\(\epsilon\)-prediction）。视觉 backbone 为 ResNet18。

## 安装

在已有 `robot_clip` 环境里补依赖（CLIP 包需已 `pip install -e ..`）：

```bash
conda activate LAD   # 或服务器上的 robot_clip 环境
pip install hydra-core omegaconf torchvision tqdm opencv-python-headless
cd /home/wny24/LAD-replication/robot_clip
pip install -e .
```

## 真机 transport 数据 → diffusion 格式

已解压的原始 zip 在：

- `data/raw_xhand/transport-test/`（XHand，单臂 `(T,19)=7` 臂 + `12` 指）
- `data/raw_g2/transport_g2_controller/`（G2，单臂 `(T,8)=7` 臂 + `1` 夹爪 `0..1000`）

转换（左右臂各写成一条 episode）：

```bash
cd /home/wny24/LAD-replication/robot_clip/diffusion
python scripts/convert_transport.py --overwrite
# 试跑前 2 条：  python scripts/convert_transport.py --max-episodes 2 --overwrite
# 只要左臂：      python scripts/convert_transport.py --sides left --overwrite
```

写出：

| 输出 key | xhand | g2 | 用途 |
|---|---|---|---|
| `action` | `(T, 12)` 指关节 | `(T, 1)` 开合 `[0,1]`（÷1000） | 进冻结 CLIP |
| `wrist_pose` | `(T, 7)` 臂关节指令 | `(T, 7)` | **不进 CLIP**，与 latent 拼接扩散 |
| `lowdim_obs` | `(T, 7)` 臂 `qpos` | `(T, 7)` | 观测条件 |
| `image` | `(T, 84, 84, 3)` | 同左 | ResNet 条件（mp4 按时间戳对齐） |

`data/manifest.yaml` 会自动写成两条等权 dataset。训练配置默认 `wrist_dim=7`、`obs.lowdim_dim=7`。

## 数据格式（通用）

每个 episode 一个 `.npz`，时间轴对齐：

| key | dtype | shape | 必填 | 含义 |
|---|---|---|---|---|
| `action` | float32 | `(T, D_raw)` | 是 | 该本体的末端动作，**物理单位**。`xhand` 为 12，`g2` 为 1，`mano` 为 189。DataLoader 会用冻结的 CLIP encoder 在线编成 32 维 latent。 |
| `wrist_pose` | float32 | `(T, D_wrist)` | `wrist_dim>0` 时必填 | **不经过 CLIP**，与 latent 拼接后作为扩散目标。transport 数据为 7 维臂关节；笛卡尔 EEF 可用 9（xyz+6D）。 |
| `image` | uint8 | `(T, H, W, 3)` 或 `(T, Ncam, H, W, 3)` | `obs.use_image=true` 时必填 | 与动作对齐的相机图。 |
| `lowdim_obs` | float32 | `(T, D_low)` | `obs.lowdim_dim>0` 时必填 | 可选本体状态。 |

`manifest.yaml` 由转换脚本生成；也可手写：

```yaml
datasets:
  - name: transport_xhand
    glob: data/xhand/*.npz
    embodiment: xhand
    weight: 1.0
  - name: transport_g2
    glob: data/g2/*.npz
    embodiment: g2
    weight: 1.0
```

多个 dataset 会按 `weight` 做 WeightedRandomSampler 混训（LAD 里各数据集等权）。`embodiment` 决定用哪路冻结 encoder。

## 训练

CLIP checkpoint 默认指向已训好的 `robot_clip/checkpoints/model_epoch_250.pth`。

```bash
conda activate LAD
cd /home/wny24/LAD-replication/robot_clip/diffusion

python train.py
```

常用 Hydra 覆盖：

```bash
python train.py \
  clip.checkpoint=/home/wny24/LAD-replication/robot_clip/checkpoints/model_epoch_250.pth \
  training.device=cuda \
  training.gpu_id=0 \
  training.batch_size=256 \
  training.steps=90000 \
  horizon=16 \
  wrist_dim=7 \
  obs.use_image=true \
  obs.lowdim_dim=7 \
  obs.n_obs_steps=2 \
  obs.n_cameras=1 \
  obs.image_size=84
```

没有图像、只有低维观测时：

```bash
python train.py obs.use_image=false obs.lowdim_dim=9
```

只在 CLIP latent 上扩散、不含腕部：

```bash
python train.py wrist_dim=0
```

checkpoint 写到 `diffusion/checkpoints/policy_step_{k}.pth` 和 `policy_latest.pth`。里面是 U-Net / 视觉编码器 / EMA / latent 归一化统计；**不含** CLIP 权重，推理时仍要加载原来的 CLIP `.pth`。

## 推理

从一条 demo 的观测采样 latent 动作，再用冻结 decoder 解回该本体：

```bash
cd /home/wny24/LAD-replication/robot_clip/diffusion

python infer.py \
  --checkpoint checkpoints/policy_latest.pth \
  --episode data/xhand/episode_0000.npz \
  --embodiment xhand \
  --t 0 \
  --ddim-steps 16 \
  --out checkpoints/sample.npz
```

`sample.npz` 含 `action`（解出的物理动作）、`wrist_pose`、`latent`、以及同窗口的 `gt_action`。换本体解码时改 `--embodiment`（例如观测来自人手、动作用 `xhand` decoder）——前提是 CLIP 的共享 latent 已经对齐。

## 训练时数据流

1. DataLoader 取出窗口：观测 `n_obs_steps` 帧，未来动作 `horizon` 步。
2. 冻结 \(q_i\)：`action` → CLIP 训练均值方差归一化 → 32 维 \(z\)。
3. 拼上 `wrist_pose`，再对 \((z, \text{wrist})\) 做一次 Gaussian 归一化（统计量存进 diffusion checkpoint）。
4. 对归一化后的序列加噪，U-Net 预测 \(\epsilon\)，MSE 只更新视觉编码器和 U-Net。
5. 推理：DDIM 去噪 → 反归一化 → 拆出 \(z\) → 冻结 \(p_j(z)\)。

## 重要配置

| 项 | 默认 | 含义 |
|---|---|---|
| `clip.checkpoint` | `../checkpoints/model_epoch_250.pth` | 冻结的 RobotCLIP。 |
| `horizon` | 16 | 预测的动作序列长度。 |
| `wrist_dim` | 9 | 与 latent 拼接的腕部位姿维数。 |
| `obs.n_obs_steps` | 2 | 条件观测帧数。 |
| `scheduler.num_train_timesteps` | 100 | DDPM 训练步数（Chi 常用 100）。 |
| `training.steps` | 90000 | 优化步数，不是 epoch。 |
| `training.batch_size` | 256 | 论文 U-Net 实验为 256。 |
| `training.ema_decay` | 0.999 | 推理用 EMA 权重。 |
