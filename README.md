# LLaVA ScienceQA: LoRA Fine-Tuning and SGLang Deployment

An end-to-end multimodal VQA pipeline for fine-tuning, evaluating, and deploying **LLaVA-v1.6-Vicuna-7B** on the image-based subset of **ScienceQA**.

The project covers ScienceQA data conversion, parameter-efficient LoRA fine-tuning, controlled Base-vs.-LoRA evaluation, model merging, SGLang serving through an OpenAI-compatible API, and a Gradio web interface.

## Highlights

- Fine-tuned LLaVA-v1.6-Vicuna-7B on **6,218 image-based ScienceQA training samples** using LoRA.
- Improved test accuracy from **56.32% to 70.10%**, an absolute gain of **13.78 percentage points**.
- Evaluated both models under an identical pipeline, including prompts, AnyRes image processing, generation settings, and answer parsing.
- Deployed the merged model with **SGLang** and exposed multimodal inference through an **OpenAI-compatible API**.
- Built a lightweight **Gradio** client for image upload, question input, and model responses.

## Results

Evaluation was performed on all **2,017 image-based ScienceQA test samples**.

| Model | Correct | Accuracy |
| --- | ---: | ---: |
| LLaVA-v1.6-Vicuna-7B Base | 1,136 / 2,017 | 56.32% |
| **ScienceQA LoRA** | **1,414 / 2,017** | **70.10%** |

**Absolute improvement:** +13.78 percentage points  
**Relative improvement:** +24.47%

## System Overview

```text
ScienceQA → Data Conversion → LoRA Fine-Tuning → Controlled Evaluation
                                                     ↓
Gradio Client ← OpenAI-Compatible API ← SGLang ← Merged Model
```

## Repository Structure

```text
.
├── scripts/
│   ├── convert_scienceqa.py
│   ├── eval_scienceqa_official.py
│   ├── finetune_lora.sh
│   └── merge_lora_weights.py
├── gradio_llava_scienceqa.py
├── 215927.png
└── README.md
```

| File | Description |
| --- | --- |
| `scripts/convert_scienceqa.py` | Filters image-based samples and converts them to LLaVA conversation format |
| `scripts/finetune_lora.sh` | Defines and launches LoRA fine-tuning |
| `scripts/eval_scienceqa_official.py` | Runs controlled Base and LoRA evaluation |
| `scripts/merge_lora_weights.py` | Merges the trained adapter into the base model |
| `gradio_llava_scienceqa.py` | Sends multimodal requests to the SGLang API |

The repository contains project-specific scripts only. Model weights, datasets, checkpoints, and the upstream LLaVA source code are not included.

## Data Preparation

ScienceQA includes both image-based and text-only questions. This project retains only samples with an associated image.

| Split | Full dataset | Image-based subset |
| --- | ---: | ---: |
| Train | 12,726 | 6,218 |
| Test | 4,241 | 2,017 |

Convert the dataset:

```bash
python scripts/convert_scienceqa.py
```

The script exports image files and creates `scienceqa_train_llava.json` and `scienceqa_test_llava.json` in LLaVA conversation format.

## LoRA Fine-Tuning

Fine-tuning adapts the Vicuna language model while keeping the CLIP vision encoder and multimodal projector frozen. LoRA adapters are applied to the attention and MLP projection layers.

| Parameter | Value |
| --- | ---: |
| Rank | 64 |
| Alpha | 128 |
| Dropout | 0.05 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.10 |
| Epochs | 2 |

Training uses BF16, gradient accumulation, gradient checkpointing, FlashAttention 2, and DeepSpeed ZeRO-2 for a single RTX 4090 (24 GB).

Configure the local model and dataset paths in `scripts/finetune_lora.sh`, then run:

```bash
conda activate llava
bash scripts/finetune_lora.sh
```

## Evaluation

The Base and LoRA models are evaluated on the same 2,017 samples with identical images, prompts, LLaVA-v1.6 AnyRes processing, conversation template, generation configuration, and answer parser. The model weights are the only experimental variable.

```bash
conda activate llava
python scripts/eval_scienceqa_official.py
```

## Merge and Deployment

Merge the trained adapter into the base model:

```bash
conda activate llava
python scripts/merge_lora_weights.py
```

Training and deployment use separate Conda environments to avoid dependency conflicts. The verified core environments are:

| | Training | Deployment |
| --- | --- | --- |
| Environment | `llava` | `sglang-deploy` |
| Python | 3.11.15 | 3.12.13 |
| PyTorch | 2.1.2+cu121 | 2.13.0+cu130 |
| Primary stack | LLaVA, PEFT, DeepSpeed | SGLang, Gradio |

Launch the merged model with SGLang:

```bash
conda activate sglang-deploy

python -m sglang.launch_server \
    --model-path /path/to/merged-model \
    --tokenizer-path llava-hf/llava-1.5-7b-hf \
    --chat-template vicuna_v1.1 \
    --host 0.0.0.0 \
    --port 30000
```

Verify the server:

```bash
curl http://127.0.0.1:30000/v1/models
```

The model is available through the multimodal endpoint `POST /v1/chat/completions`.

## Gradio Demo

The Gradio application acts as an API client and does not load model weights directly. Start it after the SGLang server is ready:

```bash
conda activate sglang-deploy
python gradio_llava_scienceqa.py
```

- SGLang API: `http://127.0.0.1:30000`
- Gradio server: `http://0.0.0.0:6006`

![Gradio web demo](./215927.png)

## Acknowledgements

Built with [LLaVA](https://github.com/haotian-liu/LLaVA), [ScienceQA](https://scienceqa.github.io/), [SGLang](https://github.com/sgl-project/sglang), and [Gradio](https://www.gradio.app/).
