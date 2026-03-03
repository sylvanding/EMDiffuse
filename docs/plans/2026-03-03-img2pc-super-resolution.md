# 图像到点云 2D 超分辨率 — 实施计划

**目标：** 构建基于扩散模型的流水线，将模糊的显微图像（WF/SIM）转换为高分辨率概率密度图，进而通过采样恢复精确点云。

**架构：** 使用 EMDiffuse 实空间扩散模型（DDPM + UNet）学习低分辨率→高分辨率的映射。完整流程：点云 CSV → 模拟图像 → 训练 Diffusion → 推理 → 采样点云 → 评估。

**技术栈：** PyTorch 2.9.1, CUDA 13.0, Python 3.11, tifffile, scipy, numpy, pandas

---

## 物理参数

| 参数 | 值 | 备注 |
|------|-----|------|
| 图像尺寸 | 1024×1024 像素 | 高分辨率密度图 |
| 像素尺寸 | 25 nm/像素 | 视场 = 25.6 µm |
| WF PSF FWHM | 300 nm (σ ≈ 5.1 px) | 宽场显微镜 |
| SIM PSF FWHM | 120 nm (σ ≈ 2.0 px) | 结构光照明 |
| STED PSF FWHM | 50 nm (σ ≈ 0.85 px) | 受激发射损耗 |
| 密度图核 σ | 1 px (25 nm) | 真值定位精度 |

## 数据流

```
点云 (CSV, nm) → 2D 投影 (忽略 z)
    ↓
概率密度图 (1024×1024, 归一化)
    ↓
模拟 WF/SIM/STED (PSF 卷积 + 噪声)
    ↓
训练对: (WF, 密度图) 或 (SIM, 密度图)
    ↓
扩散模型训练 (实空间 DDPM)
    ↓
推理: 输入图像 → 预测密度图
    ↓
采样: 密度图 → 点云 (多项式采样)
    ↓
评估 (与真值对比)
```

## 目录结构

```
/data0/djx/EMDiffuse/
├── images/                    # 转换后的图像
│   └── microtubules/
│       ├── density/           # 真值密度图 (1024×1024 TIFF)
│       ├── wf/                # 模拟宽场图像
│       ├── sim/               # 模拟 SIM 图像
│       ├── sted/              # 模拟 STED 图像
│       └── metadata.json
├── training/                  # 训练数据
│   ├── wf2density/
│   │   ├── train_wf/          # WF patch (按 EMDiffuse 格式)
│   │   └── train_gt/          # 密度图 patch
│   └── sim2density/
│       ├── train_wf/
│       └── train_gt/
├── experiments/               # 模型检查点和日志
├── sampling_test/             # 采样测试结果
└── results/                   # 推理输出
```

---

## 已完成任务

### 任务 1: 环境与清理 ✓
- [x] 更新 `requirements.txt` (Python 3.11 + PyTorch 2.9.1 + CUDA 13.0)
- [x] 删除 `3D-SR-Unet/`, `demo/`, `example/` 目录
- [x] 修复拼写错误: `emdiffuse_conifg.py` → `emdiffuse_config.py`
- [x] 启用 cuDNN 加速
- [x] 修复 pandas 兼容性 (LogTracker)
- [x] 修复 base_model.py 硬编码的 epoch 阈值

### 任务 2: 点云 → 图像转换 ✓
- [x] `scripts/utils/pointcloud.py` — 点云读写、坐标变换
- [x] `scripts/utils/imaging.py` — PSF 模拟、卷积、噪声
- [x] `scripts/convert_pointcloud.py` — 主转换脚本
- [x] 测试 3 个样本 → 确认无误
- [x] 全量转换 1024 个样本（633 秒）

### 任务 3: 训练数据准备 ✓
- [x] `scripts/prepare_training_data.py` — 裁剪 patch
- [x] `data/sr_dataset.py` — 新的 Dataset 类
- [x] `config/WF2Density.json`, `config/SIM2Density.json` 配置文件

### 任务 4: 采样与评估 ✓
- [x] `scripts/sample_from_density.py` — 密度图→点云采样
- [x] `scripts/test_sampling_pipeline.py` — 采样质量测试
- [x] `scripts/evaluate.py` — 评估指标 (MSE/PSNR/PCC/SSIM)

### 任务 5: 训练验证 ✓
- [x] Debug 模式训练测试通过
- [x] TensorBoard 日志正常保存
- [x] Loss 正常下降

### 任务 6: 文档 ✓
- [x] 中文 README
- [x] 中文计划文档

---

## 待执行任务

### 训练流程

1. **准备全量训练数据**（转换完成后）
2. **WF→密度图** 模型训练
3. **SIM→密度图** 模型训练
4. **推理**并与真值对比
5. **采样点云**并评估

### 未来扩展

- 支持线粒体、内质网等其他结构
- 调优 PSF 参数和噪声模型
- 探索 patch 大小对训练效果的影响
- 实现 DDIM 加速采样
