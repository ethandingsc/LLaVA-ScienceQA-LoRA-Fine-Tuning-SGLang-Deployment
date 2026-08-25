# LLaVA-ScienceQA: LoRA Fine-Tuning & SGLang Deployment

An end-to-end multimodal visual question answering project based on **LLaVA-v1.6-Vicuna-7B**, covering ScienceQA data processing, LoRA fine-tuning, controlled evaluation, model merging, SGLang inference serving, and a Gradio web demo.

## Highlights

* Fine-tuned **LLaVA-v1.6-Vicuna-7B** on **6,218 image-based ScienceQA training samples** using LoRA.
* Improved accuracy from **56.32% to 70.10%** on **2,017 image-based test samples**, achieving a **+13.78 percentage-point improvement**.
* Implemented the **LLaVA-v1.6 AnyRes** image processing pipeline with multi-patch inputs and original image-size propagation.
* Conducted controlled **Base vs. LoRA** evaluation using identical test data, prompts, image preprocessing, decoding configuration, and answer parsing.
* Explored LoRA rank, alpha, learning rate, warmup ratio, gradient accumulation, and training epochs.
* Merged the LoRA adapter into the base model and deployed the resulting model using **SGLang**.
* Exposed the model through an **OpenAI-compatible API** and built a **Gradio web interface** for image-based question answering.

---

## Results

| Model                     |  Correct | Test Samples |   Accuracy |
| ------------------------- | -------: | -----------: | ---------: |
| LLaVA-v1.6-Vicuna-7B Base |     1136 |         2017 |     56.32% |
| ScienceQA LoRA-v2         |     1412 |         2017 |     70.01% |
| **Best ScienceQA LoRA**   | **1414** |     **2017** | **70.10%** |

**Best model improvement over Base: +13.78 percentage points (~24.47% relative).**

Best LoRA configuration:

| Parameter     | Value |
| ------------- | ----: |
| LoRA Rank     |    64 |
| LoRA Alpha    |   128 |
| LoRA Dropout  |  0.05 |
| Learning Rate |  2e-5 |
| Warmup Ratio  |  0.10 |
| Epochs        |     2 |

---

## Project Pipeline

```text
ScienceQA
    ↓
Image-based Sample Filtering
    ↓
LLaVA Conversation Formatting
    ↓
LLaVA-v1.6-Vicuna-7B
    ↓
LoRA Supervised Fine-Tuning
    ↓
Base vs. LoRA Evaluation
    ↓
LoRA Merge
    ↓
SGLang Inference Server
    ↓
OpenAI-compatible API
    ↓
Gradio Web Demo
```

---

## Dataset

This project uses the **ScienceQA** dataset.

Since ScienceQA contains both visual and non-visual questions, only samples containing images were retained for multimodal fine-tuning and evaluation.

| Split | Original | Without Image |      Used |
| ----- | -------: | ------------: | --------: |
| Train |   12,726 |         6,508 | **6,218** |
| Test  |    4,241 |         2,224 | **2,017** |

The original dataset was converted into the LLaVA conversation format:

```json
{
  "id": "train_000000",
  "image": "train_000000.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhich of these states is farthest north?\nA. West Virginia\nB. Louisiana\nC. Arizona\nD. Oklahoma"
    },
    {
      "from": "gpt",
      "value": "A. West Virginia"
    }
  ]
}
```

Only the question, answer choices, image, and correct answer were used. ScienceQA lecture and solution fields were excluded to keep the task focused on multimodal multiple-choice VQA.

---

## LoRA Fine-Tuning

The project performs parameter-efficient fine-tuning on the Vicuna language model while keeping the visual components frozen.

```text
CLIP Vision Encoder     Frozen
        ↓
MM Projector            Frozen
        ↓
Vicuna LLM
 └── LoRA Adapters      Trainable
```

LoRA is applied to:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The Vision Encoder and MM Projector are not included in the LoRA target modules.

This design preserves the pretrained visual-language alignment while adapting the language model to the downstream ScienceQA task under a single-GPU memory constraint.

---

## LLaVA-v1.6 AnyRes Processing

The training and evaluation pipelines preserve the LLaVA-v1.6 AnyRes image-processing behavior:

```text
Input Image
    ↓
AnyRes Image Processing
    ↓
Multi-Patch Visual Input
    +
Original Image Size
    ↓
Data Collator
    ↓
images + image_sizes
    ↓
LLaVA spatial_unpad
```

The original image dimensions are propagated together with the visual tensors because LLaVA-v1.6 uses them during spatial reconstruction and unpadding.

---

## Controlled Evaluation

To ensure a fair comparison, Base and LoRA models use exactly the same:

* 2,017 ScienceQA image-based test samples
* Input images
* Prompts
* LLaVA-v1.6 AnyRes preprocessing
* Conversation template
* Generation configuration
* Answer parser

The only experimental variable is the model:

```text
LLaVA Base
    vs.
LLaVA Base + ScienceQA LoRA
```

The evaluation parser follows the ScienceQA multiple-choice answer format and reports unparseable outputs as failures rather than guessing an option.

---

## Experiment Iteration

The initial LoRA configuration used conservative optimization settings and produced limited improvement.

The training strategy was then adjusted by increasing the number of optimizer updates through:

* More training epochs
* Lower gradient accumulation
* Moderate learning-rate adjustment

Further experiments explored LoRA capacity and optimization parameters.

The best configuration achieved:

```text
Accuracy: 70.1041%
Correct:  1414 / 2017
Failed:   0
```

Increasing LoRA capacity beyond the previous configuration produced only a small additional improvement, suggesting that performance had largely stabilized around 70%.

---

## SGLang Deployment

After fine-tuning, the LoRA adapter is merged with the base LLaVA model:

```text
LLaVA Base
    +
ScienceQA LoRA
    ↓
Merged LLaVA Model
    ↓
SGLang
```

SGLang serves the merged multimodal model as an HTTP inference service with an **OpenAI-compatible API**.

Deployment includes:

* LLaVA language model
* CLIP vision tower
* MM Projector
* Tokenizer / processor
* KV Cache
* CUDA Graph
* OpenAI-compatible inference endpoint

Example endpoint:

```text
POST /v1/chat/completions
```

---

## Gradio Demo

A lightweight Gradio frontend is used to interact with the deployed model.

```text
Browser
    ↓
Gradio Web UI
    ↓
OpenAI-compatible API
    ↓
SGLang
    ↓
LLaVA-v1.6 ScienceQA Model
    ↓
Multimodal Response
```

The demo supports:

* Image upload
* Natural-language questions
* Image Base64 encoding
* SGLang API requests
* Multimodal answer display
* Inference server status checking

---

## Tech Stack

**Model & Training**

* LLaVA-v1.6-Vicuna-7B
* Vicuna-7B
* CLIP ViT-L/14-336
* LoRA / PEFT
* Hugging Face Transformers
* DeepSpeed
* FlashAttention 2
* BF16
* Gradient Checkpointing

**Inference & Deployment**

* SGLang
* OpenAI-compatible API
* Gradio

**Environment**

* NVIDIA RTX 4090 24GB
* CUDA
* PyTorch
* Linux
* Python

---

## Repository Structure

```text
llava-scienceqa-lora/
│
├── README.md
│
├── assets/
│   ├── architecture.png
│   ├── demo.png
│   └── results.png
│
├── scripts/
│   ├── prepare_scienceqa.py
│   ├── train_lora.sh
│   ├── evaluate.py
│   ├── merge_lora.py
│   └── launch_sglang.sh
│
├── demo/
│   └── gradio_app.py
│
├── results/
│   ├── experiment_results.csv
│   └── sample_predictions.json
│
├── requirements.txt
└── .gitignore
```

---

## Key Takeaways

This project demonstrates an end-to-end workflow for adapting and deploying a multimodal large language model:

1. Multimodal dataset preprocessing and LLaVA instruction formatting
2. Parameter-efficient LoRA fine-tuning under limited GPU resources
3. LLaVA-v1.6 AnyRes image-processing pipeline
4. Controlled Base vs. fine-tuned model evaluation
5. Hyperparameter experimentation and debugging
6. LoRA model merging
7. High-performance inference serving with SGLang
8. Interactive multimodal application development with Gradio

The final model improves ScienceQA image-based test accuracy from **56.32% to 70.10%**, while completing the full workflow from multimodal data preparation to an interactive deployed application.
# LLaVA-ScienceQA-LoRA-Fine-Tuning-SGLang-Deployment
