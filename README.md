# InternVL PV Distillation

第一阶段工程：使用 `OpenGVLab/InternVL3_5-4B` 提取高分辨率 RGB 遥感影像的视觉特征和光伏存在性语义判断，为后续 Student 蒸馏准备可恢复的 Teacher cache。

当前阶段不包含 DINOv3、SAM、Student 训练、LoRA 或模型权重修改。

## PV100F + PV4026 小样本蒸馏验证

`configs/kaggle_pv_small_distill.yaml` 同时支持 Kaggle 上的两个输入：

- `zh0416/pv100f`：仅包含负样本影像，加载器会在内存中生成全零图像级标签，不修改数据集。
- `zh0416/pv4026`：包含 `512×512` RGB TIFF 影像和同名的单通道 TIFF 标签。

先只检查格式和确定性抽样，不加载模型：

```bash
python scripts/05_small_distillation_test.py \
  --config configs/kaggle_pv_small_distill.yaml \
  --selection-only
```

本地只读复现时可用绝对路径覆盖数据源，输出必须写入新目录：

```bash
python scripts/05_small_distillation_test.py \
  --config configs/kaggle_pv_small_distill.yaml \
  --source-root pv4026=C:/absolute/path/to/pv4026 \
  --source-root pv100f=C:/absolute/path/to/pv100f \
  --output-dir C:/absolute/path/to/new-output \
  --selection-only
```

再提取少量 Teacher 特征并验证图像级光伏存在性：

```bash
python scripts/05_small_distillation_test.py \
  --config configs/kaggle_pv_small_distill.yaml
```

默认抽取 2 个 `PV100F` 负样本、最多 1 个 `PV4026` 空掩膜困难负样本、2 个最小非空掩膜困难正样本和 2 个覆盖面积分位点正样本。报告保存到
`/kaggle/working/outputs/reports/small_distillation_report.json`，包含 Precision、Recall、F1/Dice 和 IoU。这里的指标衡量 Teacher 的图像级光伏存在性判断，不是像素级分割指标；当前阶段只生成蒸馏缓存，不训练 Student。

## Kaggle运行

Kaggle GPU设置为 Tesla T4，并开启 Internet。安装依赖并拉取仓库：

```bash
pip install -r requirements.txt
git clone https://github.com/Zh0416/internvl-pv-distillation.git
cd internvl-pv-distillation
```

数据路径支持配置中的 `/kaggle/input/datasets/zh0416/pv-dataset-2021`，也会自动识别截图所示的 `/kaggle/working/data/raw/zh0416/pv-dataset-2021`。目录下应包含 `images/` 和 `mask/` 或 `masks/`。

如果模型权重已经放在 `/kaggle/working/internvl_weights`，先在Notebook中把配置里的 `model.name` 改为该本地目录；目录中应至少包含 `config.json`、模型权重文件和Tokenizer文件。这样不会再次下载权重：

```python
import yaml
from pathlib import Path

config_path = Path("configs/internvl3_5_teacher.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["model"]["name"] = "/kaggle/working/internvl_weights"
config["data"]["root_candidates"] = ["/kaggle/working/data"]
config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
```

```bash
python scripts/01_check_dataset.py --config configs/internvl3_5_teacher.yaml
python scripts/02_test_teacher.py --config configs/internvl3_5_teacher.yaml
python scripts/03_extract_teacher_features.py --config configs/internvl3_5_teacher.yaml --max-samples 20
```

稳定后完整运行：

```bash
python scripts/03_extract_teacher_features.py --config configs/internvl3_5_teacher.yaml --max-samples None
```

脚本逐张读取、逐张保存并实时更新 `manifest.json`。已经存在且校验通过的 `.pt` 与 `.json` 会自动跳过。`04_resume_extraction.py` 是同一提取流程的显式断点续跑入口。

## 输出

```text
outputs/
├── reports/dataset_report.json
├── splits/{train,val,test}.txt
└── teacher_cache/
    ├── features/<stem>.pt
    ├── semantic/<stem>.json
    ├── logs/{extraction.log,failed_samples.json}
    ├── manifest.json
    └── experiment_config.json
```

特征保存为 CPU `float16` 的完整官方 `model.extract_feature()` 输出，并记录实际 shape。语义置信度统一为 `0-1`。模型使用 Hugging Face `trust_remote_code=True`，加载时优先 8-bit，加载失败再尝试4-bit；推理OOM会记录、清理显存并按重试次数继续。

打包持久化输出：

```bash
python scripts/save_teacher_cache.py --input outputs/teacher_cache --output-dir outputs/packages --part-size-gb 2
```

将生成的 `teacher_cache_part_*.tar.gz` 上传为新的 Kaggle Dataset 版本，下一次运行时挂载并解压到 `outputs/teacher_cache`，即可避免重新计算。

`experiment_config.json` 记录 seed、模型名和revision、prompt、量化、dtype、GPU、图像尺寸、动态tile设置、特征shape、耗时、成功/失败数量。

50 张分组验证使用 `configs/kaggle_pv_50_distill.yaml`：20 张常规正样本、15 张最小非空掩膜困难正样本和 15 张 `PV100F` 负样本。报告同时输出总体指标和 `metrics_by_group` 分组指标。

困难小目标使用 `configs/kaggle_pv_hard_crop_distill.yaml` 和 `scripts/07_test_hard_crop_teacher.py`。训练阶段依据已有分割掩膜生成局部放大 crop，保存 InternVL 局部特征、语义概率和原图坐标；这些是蒸馏训练用的 oracle crop，不代表部署时可以获得掩膜。部署时 Student 仍需在整图上进行密集预测。

数据质量规则会排除小于 128 像素或接触 512×512 图块边界的正样本进入语义蒸馏，并把它们记录为 `semantic_distillation_weight=0`。这类样本通常是切片产生的截断标签，不能当作 InternVL 应该从整图识别出的完整目标；像素级训练可另行使用 ignore mask 处理。

50 张 Teacher 门槛测试使用 20 张普通正样本、15 张主要目标组件距切片边缘至少 32 像素的低覆盖正样本，以及 15 张按规则强边缘特征排序的易混淆负样本。普通正样本仅从覆盖率高于困难组上限的候选中均匀取样，并明确排除已选的困难正样本，保证三组互斥。Teacher 输出的 `pv_probability` 始终表示“存在光伏”的概率，避免负样本软标签反转。报告同时给出完整影像指标、困难正样本掩膜引导裁剪指标和固定负样本局部视图误报率。掩膜引导裁剪只用于训练期 Teacher 知识提取，不能用于部署推理。

有效困难目标使用至少 96×96 的正方形 crop，并按目标包围盒较长边的 2 倍保留上下文，再由 InternVL 放大到 448 输入。这样既保留道路、屋顶等负上下文，也避免 crop 重新退化成 512×512 整图。
