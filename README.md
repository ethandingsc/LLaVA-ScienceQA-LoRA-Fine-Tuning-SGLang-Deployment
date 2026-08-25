# LLaVA-ScienceQA LoRA Fine-Tuning & Deployment

基于 **LLaVA-v1.6-Vicuna-7B** 完成 ScienceQA 多模态视觉问答任务的 LoRA 微调、定量评估与推理部署。

本项目重点记录一套可复现的 LLaVA 下游任务适配流程：

```text
ScienceQA
→ Data Processing
→ LLaVA Format Conversion
→ LoRA SFT
→ Base / LoRA Evaluation
→ LoRA Merge
→ SGLang Serving
→ OpenAI-compatible API
→ Gradio Demo
```

最终在 **2017 条 ScienceQA Image Test** 上：

| Model                     |      Accuracy |
| ------------------------- | ------------: |
| LLaVA-v1.6-Vicuna-7B Base |        56.32% |
| LoRA Fine-tuned           |    **70.10%** |
| Improvement               | **+13.78 pp** |

---

## 1. Repository Structure

本仓库不包含完整 LLaVA 源码、模型权重及 ScienceQA 数据集，仅保存本项目新增或修改的核心代码。

```text
llava-scienceqa-lora/
│
├── README.md
│
├── requirements.txt
│
│
├── scripts/
│   ├── prepare_scienceqa.py       # ScienceQA → LLaVA 格式
│   ├── train_lora.sh              # LoRA 训练
│   ├── evaluate_scienceqa.py      # Base / LoRA 评估
│   ├── merge_lora.py              # LoRA Merge
│   └── launch_sglang.sh           # SGLang Server
│
├── llava_patch/
│   └── dataset.py                 # 本项目修改的 AnyRes 数据处理代码
│
├── demo/
│   └── gradio_app.py              # Gradio Web Demo
│
└── results/
    └── experiment_results.md
```

---

## 2. Environment

训练环境：

```text
GPU: RTX 4090 24GB
OS: Linux

Python: 3.11
CUDA: 12.1
PyTorch: 2.1.2
Transformers: 4.37.2
Accelerate: 0.21.0
PEFT: 0.4.0
DeepSpeed: 0.12.6
FlashAttention2: 2.5.9.post1
```

基础模型：

```text
LLaVA-v1.6-Vicuna-7B
CLIP ViT-L/14-336
```

部署环境与训练环境独立：

```text
SGLang: 0.5.18
Transformers: 5.12.1
GPU: RTX 4090 24GB
```

---

## 3. Install LLaVA

Clone official LLaVA:

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

创建环境：

```bash
conda create -n llava python=3.11
conda activate llava

pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

下载：

```text
LLaVA-v1.6-Vicuna-7B
CLIP ViT-L/14-336
ScienceQA
```

本仓库中的 `llava_patch/` 保存对官方 LLaVA 数据处理代码的修改，可将对应文件替换到 LLaVA 项目后进行训练。

---

## 4. ScienceQA Data Processing

ScienceQA 同时包含有图和无图问题。

本项目只保留：

```python
image != None
```

最终数据：

```text
Train: 6218
Test:  2017
```

运行：

```bash
python scripts/prepare_scienceqa.py
```

转换后的 LLaVA 格式：

```json
{
  "id": "train_000000",
  "image": "train_000000.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nQuestion...\nA...\nB...\nC...\nD..."
    },
    {
      "from": "gpt",
      "value": "A. Answer"
    }
  ]
}
```

同时将 Hugging Face Parquet 中的 image bytes 导出为本地图片。

---

## 5. LoRA Fine-Tuning

训练策略：

```text
CLIP Vision Encoder     Frozen
MM Projector            Frozen
Vicuna LLM              LoRA
```

LoRA target modules：

```text
q_proj / k_proj / v_proj / o_proj
gate_proj / up_proj / down_proj
```

最终配置：

```text
LoRA r          = 64
LoRA alpha      = 128
LoRA dropout    = 0.05
Learning rate   = 2e-5
Warmup ratio    = 0.10
Epoch           = 2
```

启动训练：

```bash
bash scripts/train_lora.sh
```

其中训练脚本包含完整的：

```text
model path
data path
image folder
vision tower
LoRA config
batch size
gradient accumulation
DeepSpeed
BF16
gradient checkpointing
```

---

## 6. LLaVA-v1.6 AnyRes

训练数据处理保持 LLaVA-v1.6 AnyRes pipeline：

```text
Image
→ process_images()
→ Multi-Patch
→ image_sizes
→ DataCollator
→ LLaVA spatial_unpad
```

Smoke Test：

```text
Single Image:
[5, 3, 336, 336]

Batch:
[2, 5, 3, 336, 336]
```

相关修改代码：

```text
llava_patch/dataset.py
```

---

## 7. Evaluation

Base 与 LoRA 使用完全相同的：

```text
Test Set
Image
Prompt
AnyRes Processing
Conversation Template
Generation Config
Answer Parser
```

运行 Base：

```bash
python scripts/evaluate_scienceqa.py \
    --model_path /path/to/llava-v1.6-vicuna-7b
```

运行 LoRA：

```bash
python scripts/evaluate_scienceqa.py \
    --model_path /path/to/lora \
    --model_base /path/to/llava-v1.6-vicuna-7b
```

结果：

```text
Base:
1136 / 2017
Accuracy = 56.3213%

Best LoRA:
1414 / 2017
Accuracy = 70.1041%

Improvement:
+13.78 percentage points
```

---

## 8. Merge LoRA

训练完成后将：

```text
Base Model + LoRA Adapter
```

合并为完整模型：

```bash
python scripts/merge_lora.py \
    --model_base /path/to/llava-v1.6-vicuna-7b \
    --model_path /path/to/lora \
    --save_path /path/to/merged-model
```

得到：

```text
Merged LLaVA
```

用于后续 SGLang 部署。

---

## 9. SGLang Deployment

创建独立部署环境并安装：

```bash
conda create -n sglang-deploy python=3.11
conda activate sglang-deploy

pip install "sglang[all]"
```

启动服务：

```bash
bash scripts/launch_sglang.sh
```

核心启动方式：

```bash
python -m sglang.launch_server \
    --model-path /path/to/merged-model \
    --tokenizer-path llava-hf/llava-1.5-7b-hf \
    --chat-template vicuna_v1.1 \
    --host 0.0.0.0 \
    --port 30000
```

检查服务：

```bash
curl http://127.0.0.1:30000/v1/models
```

推理接口：

```text
POST /v1/chat/completions
```

---

## 10. Gradio Demo

Gradio 作为前端调用 SGLang API：

```text
Image + Question
       ↓
     Gradio
       ↓
Base64 Image
       ↓
POST /v1/chat/completions
       ↓
     SGLang
       ↓
      LLaVA
       ↓
     Answer
```

启动：

```bash
python demo/gradio_app.py
```

默认：

```text
SGLang: http://127.0.0.1:30000
Gradio: http://0.0.0.0:6006
```

---

## 11. Final Result

```text
ScienceQA Image Train: 6218
ScienceQA Image Test:  2017

LLaVA-v1.6 Base:       56.32%
Best LoRA:             70.10%
Absolute Improvement:  +13.78 pp
```

完整流程：

```text
Data Processing
→ LoRA Fine-Tuning
→ Controlled Evaluation
→ Model Merge
→ SGLang API
→ Gradio Demo
```
