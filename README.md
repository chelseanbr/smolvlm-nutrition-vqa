# csml-final-project
## Setup

```bash
conda create --name csml-proj python=3.12 pip -y

conda activate csml-proj

pip install -r requirements.txt
```

## Data
```bash
tar -czvf data.tar.gz data
```
```bash
tar -xzvf data.tar.gz
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

Final Test:
Benchmark:
```bash
python -m src.finetune test vlm_model/SmolVLM-256M-Instruct HuggingFaceTB/SmolVLM-256M-Instruct```