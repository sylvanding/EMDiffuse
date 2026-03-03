# DiffusePoint: 基于扩散模型的 2D 超分辨率点云重建

基于扩散模型的框架，将低分辨率显微图像（WF/SIM）转换为高分辨率概率密度图，从而实现精确的点云重建。

## 原理

### 实空间扩散超分辨

本项目使用 **去噪扩散概率模型 (DDPM)** 在 **实像素空间**（非潜空间）进行学习，将低分辨率显微图像映射到高分辨率概率密度图。

**前向过程：** 训练时，对真值密度图 $y_0$ 逐步添加高斯噪声，经过 $T$ 个时间步：

$$y_t = \sqrt{\bar{\alpha}_t} \cdot y_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**反向过程：** UNet $\epsilon_\theta$ 学习从带噪图像 $y_t$、条件输入（WF/SIM 图像）和噪声水平 $\bar{\alpha}_t$ 预测噪声 $\epsilon$：

$$\hat{\epsilon} = \epsilon_\theta([y_{cond}, y_t], \bar{\alpha}_t)$$

**推理：** 从纯高斯噪声出发，模型以输入显微图像为条件，逐步去噪生成清晰的密度图。每一步反向采样：

$$y_{t-1} = \mu_\theta(y_t, t) + \sigma_t \cdot z, \quad z \sim \mathcal{N}(0, I)$$

**为什么选择实空间而非潜空间？** 潜空间（如 VAE 编码）会在编解码过程中丢失精细结构信息。对于微管等亚像素级精细结构，实空间扩散可以保留每个像素的物理信息，避免压缩损失。

### 流水线概览

```
点云 (CSV, nm 坐标)
    |
    v
2D 投影 (xy 平面, 忽略 z)
    |
    v
模拟显微图像:
    - WF   (宽场, PSF FWHM = 300nm)
    - SIM  (结构光照明, PSF FWHM = 120nm)
    - STED (受激发射损耗, PSF FWHM = 50nm)
    - 概率密度图 (真值, sigma = 25nm)
    |
    v
扩散模型训练:
    - WF  -> 密度图
    - SIM -> 密度图
    |
    v
推理: 输入图像 -> 预测密度图
    |
    v
点云采样 (从密度图多项式采样)
    |
    v
评估 (与真值对比)
```

### 采样原理

密度图代表每个像素处的分子概率密度。采样过程：
1. 将密度图展平为一维概率分布（归一化至和为 1）
2. 使用多项式采样选取 N 个像素位置
3. 在每个选中像素内添加均匀随机偏移（亚像素精度）
4. 将像素坐标转换回纳米坐标

这使得从单张密度图可以采样 **任意数量** 的点。

## 环境配置

测试环境：Ubuntu + NVIDIA RTX 5090, Python 3.11, CUDA 13.0

```bash
conda create -n emdiffuse python=3.11 -y
conda activate emdiffuse
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## 快速开始

### 第一步：点云转图像

将点云 CSV 转换为模拟显微图像（WF/SIM/STED）和真值概率密度图。

```bash
# 先测试 3 个样本（附带可视化）：
python scripts/convert_pointcloud.py \
    --input_dir /data0/djx/img2pc_2d/microtubules \
    --output_dir /data0/djx/EMDiffuse/images/microtubules \
    --samples 3 --visualize

# 转换全部 1024 个样本：
python scripts/convert_pointcloud.py \
    --input_dir /data0/djx/img2pc_2d/microtubules \
    --output_dir /data0/djx/EMDiffuse/images/microtubules \
    --samples all
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image_size` | 1024 | 输出图像尺寸（像素） |
| `--pixel_size` | 25.0 | 像素尺寸（nm） |
| `--modalities` | 全部 | 生成的模态（wf/sim/sted/density） |
| `--structure_type` | microtubules | 生物结构类型 |
| `--visualize` | 关闭 | 生成对比可视化图 |

### 第二步：准备训练数据

将图像对裁剪为 patch 用于扩散模型训练。

```bash
# WF -> 密度图 训练数据：
python scripts/prepare_training_data.py \
    --image_dir /data0/djx/EMDiffuse/images/microtubules \
    --output_dir /data0/djx/EMDiffuse/training/wf2density \
    --input_modality wf --target_modality density \
    --patch_size 256 --overlap 0.125 --train_ratio 0.9

# SIM -> 密度图 训练数据：
python scripts/prepare_training_data.py \
    --image_dir /data0/djx/EMDiffuse/images/microtubules \
    --output_dir /data0/djx/EMDiffuse/training/sim2density \
    --input_modality sim --target_modality density \
    --patch_size 256 --overlap 0.125 --train_ratio 0.9
```

### 第三步：训练扩散模型

```bash
# 训练 WF -> 密度图 模型（多 GPU）：
python run.py -c config/WF2Density.json -b 8 --gpu 0,1,2,3 --port 20022 \
    --path /data0/djx/EMDiffuse/training/wf2density/train_wf --lr 5e-5

# 训练 SIM -> 密度图 模型：
python run.py -c config/SIM2Density.json -b 8 --gpu 0,1,2,3 --port 20023 \
    --path /data0/djx/EMDiffuse/training/sim2density/train_wf --lr 5e-5
```

训练检查点和日志保存在 `/data0/djx/EMDiffuse/experiments/`。

**监控训练：** 使用 TensorBoard 实时监控 loss 和验证结果：

```bash
tensorboard --logdir /data0/djx/EMDiffuse/experiments/ --port 6006
```

浏览器打开 `http://localhost:6006` 即可看到：
- 训练 loss 曲线（train/mse_loss）
- 验证指标（val/mae）
- 验证图像对比（GT vs 预测 vs 输入）

### 第四步：推理

```bash
python run.py -p test -c config/WF2Density.json --gpu 0 -b 8 \
    --path /data0/djx/EMDiffuse/training/wf2density/test_wf \
    --resume /data0/djx/EMDiffuse/experiments/WF2Density/best \
    --mean 1 --step 1000
```

| 参数 | 说明 |
|------|------|
| `--mean` | 生成并平均的采样次数（更多=更稳定） |
| `--step` | 扩散步数（更多=更高质量，默认 1000） |
| `--resume` | 模型权重路径 |

### 第五步：从密度图采样点云

```bash
# 从预测密度图采样指定数量的点：
python scripts/sample_from_density.py \
    --density_map /data0/djx/EMDiffuse/results/density/0001.tif \
    --n_points 50000 \
    --output /data0/djx/EMDiffuse/results/sampled/0001.csv \
    --metadata /data0/djx/EMDiffuse/images/microtubules/metadata.json \
    --sample_id 0001 --visualize
```

### 第六步：测试采样质量

验证采样流程：密度图 → 采样 → 重建密度图 → 对比。

```bash
python scripts/test_sampling_pipeline.py \
    --image_dir /data0/djx/EMDiffuse/images/microtubules \
    --output_dir /data0/djx/EMDiffuse/sampling_test \
    --n_points 50000 100000 200000 400000 \
    --samples 3
```

### 第七步：评估结果

```bash
# 对比预测密度图与真值：
python scripts/evaluate.py \
    --pred_density /data0/djx/EMDiffuse/results/density/0001.tif \
    --gt_density /data0/djx/EMDiffuse/images/microtubules/density/0001.tif \
    --output_dir /data0/djx/EMDiffuse/results/evaluation \
    --visualize
```

## 项目结构

```
EMDiffuse/
├── config/                          # 模型配置
│   ├── WF2Density.json              # 宽场 -> 密度图
│   ├── SIM2Density.json             # SIM -> 密度图
│   ├── EMDiffuse-n.json             # (旧版) EM 去噪
│   └── EMDiffuse-r.json             # (旧版) EM 超分辨
├── scripts/                         # 数据处理流水线
│   ├── convert_pointcloud.py        # 点云 -> 图像
│   ├── prepare_training_data.py     # 裁剪 patch 用于训练
│   ├── sample_from_density.py       # 密度图 -> 点云采样
│   ├── test_sampling_pipeline.py    # 采样质量测试
│   ├── evaluate.py                  # 质量评估
│   └── utils/
│       ├── imaging.py               # PSF 模拟、噪声、密度图
│       └── pointcloud.py            # 点云读写、坐标变换
├── models/                          # 扩散模型
│   ├── EMDiffuse_model.py           # 训练/推理逻辑 (DiReP)
│   ├── EMDiffuse_network.py         # DDPM 网络（前向/反向）
│   └── guided_diffusion_modules/
│       └── unet.py                  # UNet 骨干网络
├── data/
│   ├── dataset.py                   # 原始数据集类
│   └── sr_dataset.py                # 超分辨数据集
├── core/                            # 训练基础设施
│   ├── base_model.py                # 训练循环
│   ├── base_network.py              # 权重初始化
│   ├── praser.py                    # 配置解析
│   └── logger.py                    # 日志与 TensorBoard
├── run.py                           # 主入口
├── requirements.txt
└── docs/plans/                      # 实施计划
```

## 成像参数

| 模态 | PSF FWHM (nm) | σ (像素 @ 25nm/px) | 说明 |
|------|:------------:|:------------------:|------|
| WF | 300 | 5.1 | 宽场——衍射极限 |
| SIM | 120 | 2.0 | 结构光照明——超越衍射极限约 2 倍 |
| STED | 50 | 0.85 | 受激发射损耗——亚衍射极限 |
| Density | 25 | 1.0 | 真值概率密度 |

## 数据格式

### 输入：点云 CSV
```
x [nm],y [nm],z [nm]
22731.510,10357.466,55.271
15153.398,836.224,1205.677
...
```

### 输出：TIFF 图像
- 16 位无符号整数 (0-65535)
- 1024×1024 像素，25 nm/像素
- 视场：25.6 µm × 25.6 µm

## 可扩展性

支持不同生物结构。添加新结构：

1. 将点云 CSV 放入 `{structure}_{id}_{count}k/` 格式的文件夹
2. 转换时设置 `--structure_type` 和 `--pattern`
3. 按需调整 `scripts/utils/imaging.py` 中的 PSF/噪声参数
4. 基于 `WF2Density.json` 创建新配置

已支持/计划支持的结构：
- 微管 (Microtubules) ✓
- 线粒体 (Mitochondria) — 待添加
- 内质网 (Endoplasmic Reticulum) — 待添加

## 致谢

基于 [EMDiffuse](https://github.com/Luchixiang/EMDiffuse) 框架（Nature Communications, 2024）改造，用于 2D 超分辨显微与点云重建。
