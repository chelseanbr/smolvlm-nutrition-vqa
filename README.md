# VQA for Bioavailable Iron using Small VLMs

## Overview
Accurate nutritional estimation is critical for monitoring diet and alleviating iron deficiencies. Current systems only estimate total iron rather than bioavailable iron or its individual factors (ingredient portions, cooking methods, and iron-inhibiting micronutrients like calcium). This project explores how Vision-Language Models (VLMs) can bridge this gap.

## Approach
- **Data:** Created a novel Visual Question Answering (VQA) dataset by supplementing a subset of MM-Food-100K with measurements from the USDA FoodData Central API.

- **Models:** Fine-tuned small Vision-Language Models from the SmolVLM family.

- **Fine-Tuning:** Performed Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA) for efficient training.

## Key Results
The final fine-tuned 2.2B SmolVLM-Instruct model established a baseline for using VLMs to predict key nutritional components from real-world images, achieving:

- **Total Classification Accuracy:** 0.587

- **Mean Absolute Error (MAE) for Iron:** 1.45 mg

---

➡️ Full paper: https://doi.org/10.31224/6975

## Setup

```bash
conda create --name csml-proj python=3.12 pip -y

conda activate csml-proj

pip install -r requirements.txt
```

## Finetuning
### Demo Data

Train:
```bash
python -m src.finetune demo_train HuggingFaceTB/SmolVLM-256M-Instruct
```

Benchmark:
```bash
!python -m src.finetune val demo_train/SmolVLM-256M-Instruct HuggingFaceTB/SmolVLM-256M-Instruct
```

### Full Data

Train:
```bash
python -m src.finetune train HuggingFaceTB/SmolVLM-256M-Instruct
```

Benchmark:
```bash
python -m src.finetune val vlm_model/SmolVLM-256M-Instruct HuggingFaceTB/SmolVLM-256M-Instruct
```

Final Test - Benchmark:
```bash
python -m src.finetune test vlm_model/SmolVLM-256M-Instruct HuggingFaceTB/SmolVLM-256M-Instruct
```
