from pathlib import Path

import functools
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import VQADataset, benchmark, benchmark_extended, compute_metrics_test, MACRO_LABELS, MICRO_LABELS, ALL_LABELS, NUM_VALUES

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

# processor = AutoProcessor.from_pretrained(DEFAULT_MODEL)
processor = None
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def load(ckpt_name: str, model_name: str = "vlm_model") -> BaseVLM:
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM(checkpoint=ckpt_name)
    vlm.model = PeftModel.from_pretrained(vlm.model, model_path).to(vlm.device)
    vlm.model.eval()

    return vlm


def custom_data_collator(features: list[dict[str, torch.Tensor]], processor) -> dict[str, torch.Tensor]:
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }


class VQADatasetForTraining(Dataset):
    def __init__(self, dataset: VQADataset, processor: AutoProcessor, ckpt_name: str):
        self.dataset = dataset
        self.processor = processor
        self.features = ["image", "question", "answer"]

        # --- START FIX FOR IMAGE TOKEN LOOKUP ---        
        if 'Qwen' in ckpt_name:
            # Qwen-VL models use the specific token: <|image|>
            image_token_string = "<|image|>"
        elif 'SmolVLM' in ckpt_name:
            # SmolVLM uses the basic LLaVA token: <image>
            image_token_string = "<image>"
        else:
            # Fallback for unexpected models
            image_token_string = "<image>"

        # Find the token ID using the dictionary lookup, which is safer than index lookup
        # We assume the token string is one of the additional special tokens
        if image_token_string in self.processor.tokenizer.get_vocab():
             self.image_token_id = self.processor.tokenizer.get_vocab()[image_token_string]
        elif image_token_string in self.processor.tokenizer.additional_special_tokens:
             # Look up in the additional special tokens IDs list (your original method)
             self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                 self.processor.tokenizer.additional_special_tokens.index(image_token_string)
             ]
        else:
            raise ValueError(f"Image token '{image_token_string}' not found in tokenizer vocabulary for {ckpt_name}. Check model documentation.")
        
        # --- END FIX ---
        
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        # # Define a safe maximum pixel dimension (e.g., 2000x2000)
        # MAX_DIMENSION = 2000
        # MAX_SIZE = (MAX_DIMENSION, MAX_DIMENSION) 

        # # Check if the image size exceeds the limit AND is large enough to bother resizing
        # if image.size[0] > MAX_DIMENSION or image.size[1] > MAX_DIMENSION:
        #     # Use a high quality filter (LANCZOS) when downscaling
        #     image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)

        # Prepare input text in chat format
        input_message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["question"]}]}]
        prompt = self.processor.apply_chat_template(input_message, add_generation_prompt=True)
        full_text = prompt + str(item["answer"])  # append the answer to the prompt

        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Get answer length
        answer_ids = self.processor(
            images=None, text=str(item["answer"]), return_tensors="pt", truncation=True
        ).input_ids.squeeze(0)
        answer_len = len(answer_ids)

        # Prepare labels: mask everything except the answer tokens
        labels = input_ids.clone()
        labels[:-answer_len] = -100  # only keep loss on answer

        # Ensure EOS token is at the end of the sequence
        if input_ids[-1] != self.processor.tokenizer.eos_token_id:
            input_ids = torch.cat([input_ids, torch.tensor([self.processor.tokenizer.eos_token_id])])
            attention_mask = torch.cat([attention_mask, torch.tensor([1])])
            labels = torch.cat([labels, torch.tensor([self.processor.tokenizer.eos_token_id])])

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.long(),
        }


def train(
    ckpt_name: str,
    num_train_epochs: int = 0.05,  # use only 0.05 epoch for training
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 8,
    lora_alpha: int = 32,
    eval_steps=50,       
    data_dir: Path | None = None,
    train_dataset_name: str = "train-grader",
    val_dataset_name: str = "val-grader",
    # output_dir: str = "vlm_sft",
    # output_dir: str = "homework/vlm_model",
    output_dir: str = "src/vlm_model",
    learning_rate: float = 5e-4,
    lora_dropout: float = 0.0,
    num_workers: int = 12,
    evaluation_strategy="steps", # Switch to "steps" to evaluate frequently
    load_best_model_at_end=True, # Load best model at end based on eval metric
    logging_steps=50
):
    """
    Fine-tune a VLM model using LoRA.

    Args:
        model_name: Name of the base model to fine-tune
        data_dir: Directory containing the dataset
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    """
    print(f'{ckpt_name=}')
    global processor
    processor = AutoProcessor.from_pretrained(ckpt_name)

    vlm = BaseVLM(checkpoint=ckpt_name)

    # Create output directory
    out_name = ckpt_name.split('/')[-1]
    output_dir = f'{output_dir}/{out_name}'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    processor = vlm.processor
    model = vlm.model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.train()

    # Prepare datasets
    train_dataset = VQADataset(train_dataset_name, data_dir)
    val_dataset_raw = VQADataset(val_dataset_name, data_dir) # Keep raw dataset
    # val_questions = [item["question"] for item in val_dataset_raw] # Extract questions

    train_dataset = VQADatasetForTraining(train_dataset, processor, ckpt_name)
    val_dataset_processed = VQADatasetForTraining(val_dataset_raw, processor, ckpt_name)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
        # metric_for_best_model="eval_iron_mg_mae",
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower is better
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        # per_device_eval_batch_size=4,
        per_device_eval_batch_size=256,
        eval_accumulation_steps=1,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # # Define the partial function to inject the required data
    # metric_fn = functools.partial(
    #     compute_metrics, 
    #     processor=processor, 
    #     val_questions=val_questions
    # )

    collator_fn = functools.partial(custom_data_collator, processor=processor)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_processed,
        data_collator=collator_fn,
        # compute_metrics=metric_fn,
    )

    # model.config.max_new_tokens = 48

    # Train the model
    trainer.train()

    # --- MANUAL LOGGING OF ALL METRICS FROM HISTORY ---
    # This loop is designed to capture all logged metrics (train and eval)
    
    # NOTE: The step number (i) should actually be taken from the 'step' key 
    #       in the log dictionary, not the enumerate index.
    
    for log in trainer.state.log_history:
        # Check for the step number
        step = log.get("step")
        if step is None:
            # If 'step' is missing (e.g., initial logs), skip or use a counter
            continue 

        for key, value in log.items():
            # 1. Log Training Loss
            if key == "loss":
                writer.add_scalar("Loss/train", value, step)
            
            # 2. Log Evaluation Metrics (includes eval_loss and all your custom MAE/RMSE)
            elif key.startswith("eval_"):
                # Clean up the key name for better grouping in TensorBoard if desired,
                # but using the full key name is safer.
                writer.add_scalar(f"Evaluation/{key}", value, step)

    # Save the model
    trainer.save_model(output_dir)

    # Close TensorBoard writer
    writer.close()

    return model, processor


def evaluate(model: nn.Module, val_loader: DataLoader) -> float:
    """
    Evaluate the model on the validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            val_loss += outputs.loss.item()

    model.train()
    return val_loss / len(val_loader)


def compute_metrics(eval_pred, processor, val_questions) -> dict[str, float]:
    tokenizer = processor.tokenizer
    
    # eval_pred is a tuple (predictions, labels)
    predictions, labels = eval_pred

    # --- FIX: ROBUST PREDICTION EXTRACTION ---
    # 1. Handle the case where predictions is a tuple (the source of the error)
    if isinstance(predictions, tuple) or isinstance(predictions, list):
        predictions = predictions[0] # Take the first element, which should be the core tensor/array
    
    # 2. Ensure data is a NumPy array for consistent processing
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
        
    # If the predictions are logits (3D), take argmax to get token IDs (2D)
    # This check now runs safely after we've confirmed 'predictions' is a NumPy array.
    if predictions.ndim > 2:
        predictions = np.argmax(predictions, axis=-1)

    # Process labels (remains the same)
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Cast to int64 for safe tokenization/decoding
    labels = labels.astype(np.int64)
    predictions = predictions.astype(np.int64)

    # Replace the ignore index (-100) with the pad token ID for decoding
    labels[labels == -100] = tokenizer.pad_token_id 
    
    # Batch decode
    pred_strs = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # --- Metric Storage Initialization ---
    true_regression_values = [[] for _ in range(NUM_VALUES)]
    pred_regression_values = [[] for _ in range(NUM_VALUES)]
    true_class_labels = []
    pred_class_labels = []
    metrics = {}
    
    # --- PARSING LOOP ---
    for sample_index, (pred_str, label_str) in enumerate(zip(pred_strs, label_strs)):
        
        clean_label = label_str.strip().lower()
        clean_pred = pred_str.strip().lower()

        # Check if it's a regression output (contains a comma)
        if ',' in clean_label:
            # --- REGRESSION TASK LOGIC ---
            
            # 1. Parse GROUND TRUTH (must succeed for sample to be valid)
            try:
                true_list = [float(x.strip()) for x in clean_label.split(',')]
            except Exception:
                continue # Skip if ground truth itself is malformed
            
            # 2. Parse PREDICTION (must handle failure by defaulting to zero)
            try:
                # Attempt to parse the prediction normally
                pred_list = [float(x.strip()) for x in clean_pred.split(',')]
                
                # Check for correct length, which is a common failure point
                if len(pred_list) != len(true_list):
                    raise ValueError("Prediction list length mismatch.")

            except Exception:
                # If parsing or length check fails, set predicted values to 0.0
                pred_list = [0.0] * len(true_list)
            
            # 3. Identify Task and Append Values
            try:
                question = val_questions[sample_index]
                q_lower = question.lower()
                
                if "calories" in q_lower or "fat" in q_lower:
                    base_index = 0
                elif "total weight" in q_lower or "iron" in q_lower:
                    base_index = 4
                else:
                    # If question is ambiguous, skip appending, but only if its a regression sample
                    continue 

                for i in range(len(true_list)):
                    true_regression_values[base_index + i].append(true_list[i])
                    pred_regression_values[base_index + i].append(pred_list[i])
            
            except Exception:
                # Skip if index access or question handling fails
                continue
            
        else:
            # --- CLASSIFICATION TASK LOGIC (remains the same) ---
            true_class_labels.append(clean_label)
            answer_len = len(clean_label)
            
            if clean_pred[:answer_len] == clean_label:
                pred_class_labels.append(clean_label) 
            else:
                pred_class_labels.append(clean_pred)

    # --- Calculation Loop (Now guaranteed to run if val set has regression samples) ---
    for i, label_name in enumerate(ALL_LABELS):
        true_values = np.array(true_regression_values[i])
        pred_values = np.array(pred_regression_values[i])
        
        if len(true_values) > 0:
            mae = mean_absolute_error(true_values, pred_values)
            rmse = np.sqrt(mean_squared_error(true_values, pred_values))
            
            metrics[f"eval_{label_name}_mae"] = mae
            metrics[f"eval_{label_name}_rmse"] = rmse

    # --- Classification Metrics ---
    if true_class_labels:
        correct_count = sum(1 for t, p in zip(true_class_labels, pred_class_labels) if t == p)
        total_samples = len(true_class_labels)
        metrics["eval_classification_accuracy"] = correct_count / total_samples
        
    return metrics


def demo_train(ckpt_name: str):
    train(
        ckpt_name=ckpt_name,
        train_dataset_name="train_demo",
        val_dataset_name="val_demo",
        # output_dir="demo_train",
        # output_dir="homework/demo_train",
        output_dir="src/demo_train",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
        eval_steps=4,
        logging_steps=1,
    )


def val_model(ckpt_path: str, ckpt_name: str, max_samples: int = 256, val_dataset: str = "val-grader"):
    print(f'{ckpt_name=}')
    global processor
    processor = AutoProcessor.from_pretrained(ckpt_name)

    valset = VQADataset(val_dataset)

    llm = load(ckpt_name, ckpt_path)

    benchmark_result = benchmark(llm, valset, max_samples)
    print(benchmark_result.accuracy)


def test_model(ckpt_path: str, ckpt_name: str, max_samples: int = 256, test_dataset: str = "test-grader"):
    print(f'{ckpt_name=}')
    global processor
    processor = AutoProcessor.from_pretrained(ckpt_name)

    testset = VQADataset(test_dataset)

    llm = load(ckpt_name, ckpt_path)

    benchmark_result = benchmark(llm, testset, max_samples)
    print(benchmark_result.accuracy)


def get_ext_benchmark(ckpt_path: str, ckpt_name: str, dataset: str = "val-grader", 
                      mini_batch_size: int = 32, max_samples = None):
    print(f'{ckpt_name=}')
    global processor
    processor = AutoProcessor.from_pretrained(ckpt_name)

    testset = VQADataset(dataset)

    llm = load(ckpt_name, ckpt_path)

    # The benchmark_result object now includes the questions and answers/responses
    # benchmark_result_with_data = benchmark(llm, testset, max_samples)
    benchmark_result_with_data = benchmark_extended(llm, testset, mini_batch_size, max_samples)

    # Extract required lists from the benchmark result and dataset
    responses = [s.model_answer for s in benchmark_result_with_data.samples]
    gt_answers = [s.correct_answer for s in benchmark_result_with_data.samples]
    gt_questions = [s.question for s in benchmark_result_with_data.samples]
    
    # --- CALCULATE ALL METRICS ---
    final_metrics = compute_metrics_test(responses, gt_answers, gt_questions)
    
    print("\n" + "="*50)
    print(f"      FINAL BENCHMARK RESULTS ({dataset.upper()} SET)")
    print("="*50)
    
    # --- NEW SECTION: CLASSIFICATION ACCURACY ---
    
    # 1. Print Total Classification Accuracy
    total_accuracy_strict = final_metrics.get("total_accuracy_strict", 0.0)
    total_accuracy_fuzzy = final_metrics.get("total_accuracy_fuzzy", 0.0)
    
    print("CLASSIFICATION ACCURACY:")
    print(f"  {'TOTAL ACCURACY (Strict)':<25} | {total_accuracy_strict:.4f}")
    print(f"  {'TOTAL ACCURACY (Fuzzy 90)':<25} | {total_accuracy_fuzzy:.4f}") # <-- NEW Printout
    
    # 2. Print Per-Type Classification Accuracy
    classification_types = ['dish_name', 'category', 'cooking_method']
    
    for q_type in classification_types:
        display_name = q_type.replace('_', ' ').upper()
        
        accuracy_strict = final_metrics.get(f"accuracy_{q_type}_strict")
        accuracy_fuzzy = final_metrics.get(f"accuracy_{q_type}_fuzzy") # <-- NEW Retrieval
        
        if accuracy_strict is not None:
            print(f"  {display_name + ' ACCURACY (Strict)':<25} | {accuracy_strict:.4f}")
        if accuracy_fuzzy is not None:
            print(f"  {display_name + ' ACCURACY (Fuzzy 90)':<25} | {accuracy_fuzzy:.4f}") # <-- NEW Printout

    # Print Regression Scores
    # print("\nREGRESSION SCORES (MAE/RMSE/MAPE):") # Updated Header
    print("\nREGRESSION SCORES (MAE/RMSE):") # Updated Header
    for label in ALL_LABELS:
        mae = final_metrics.get(f"{label}_mae")
        rmse = final_metrics.get(f"{label}_rmse")
        # # --- ADD MAPE RETRIEVAL ---
        # mape = final_metrics.get(f"{label}_mape")
        
        # Check for MAE/RMSE to be present
        if mae is not None and rmse is not None:
            # --- UPDATED PRINT FORMAT ---
            print(f"  {label.upper():<15} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"demo_train": demo_train, "train": train, "val": val_model, "test": test_model, "benchmark": get_ext_benchmark})
