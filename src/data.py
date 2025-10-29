from http.client import responses
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from dataclasses import dataclass, field
from fuzzywuzzy import fuzz
from pathlib import Path
from typing import Any, List, Dict

DATA_DIR = Path(__file__).parent.parent / "data"
# Define the fixed order and labels outside the function
MACRO_LABELS = ["calories_kcal", "fat_g", "protein_g", "carbs_g"]
MICRO_LABELS = ["total_weight_g", "iron_mg", "calcium_mg", "vitamin_C_mg"] 
ALL_LABELS = MACRO_LABELS + MICRO_LABELS
NUM_VALUES = len(ALL_LABELS) # Should be 8


class VQADataset:
    def __init__(self, split: str, data_dir: Path = None, max_samples: int = None):
        """
        Initialize the VQA dataset.

        Args:
            split: Dataset split ('train', 'valid_grader', 'train_demo')
            data_dir: Directory containing the dataset (default: DATA_DIR)
        """
        self.data_dir = data_dir or DATA_DIR

        # Load all QA pairs for the split
        self.qa_pairs = []

        # Find all QA pair files for the split
        qa_files = list(self.data_dir.glob(f"{split}/*_qa_pairs.json"))

        for qa_file in qa_files:
            with open(qa_file) as f:
                qa_pairs = json.load(f)
                self.qa_pairs.extend(qa_pairs)

        if max_samples is not None:
            self.qa_pairs = self.qa_pairs[:max_samples]

        print(f"Loaded {len(self.qa_pairs)} QA pairs for {split} split")

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a QA pair by index.

        Args:
            idx: Index of the QA pair

        Returns:
            Dictionary containing the QA pair and image path
        """
        qa_pair = self.qa_pairs[idx]

        # Construct the full path to the image
        image_path = os.path.join(self.data_dir, qa_pair["image_file"])

        return {
            "image_path": image_path,
            "question": qa_pair["question"],
            "answer": str(qa_pair["answer"]),
        }


@dataclass
class VQABenchmarkResult:
    @dataclass
    class Sample:
        image_path: str
        question: str
        model_answer: str
        correct_answer: str
        is_correct: bool

    # --- Updated Fields ---
    accuracy: float = 0.0
    
    # Store all per-value MAE/RMSE scores here
    regression_metrics: Dict[str, float] = field(default_factory=dict) 
    
    samples: List[Sample] = field(default_factory=list)

    @classmethod
    def from_answers(
        cls, answers: list[str], gt_dataset: list[dict[str, Any]], max_samples: int = None
    ) -> "VQABenchmarkResult":
        """
        Create a benchmark result from model answers.

        Args:
            answers: List of model answers
            dataset: Dataset used for evaluation

        Returns:
            Benchmark result
        """
        samples = []
        correct_count = 0

        if max_samples is None:
            max_samples = min(len(answers), len(gt_dataset))
        else:
            max_samples = len(gt_dataset)

        for i in range(max_samples):
            item = gt_dataset[i]
            answer = answers[i]

            # For string answers, we use exact matching
            answer_len = len(item["answer"].strip())
            is_correct = answer.strip().lower()[:answer_len] == item["answer"].strip().lower()[:answer_len]
            samples.append(
                cls.Sample(
                    image_path=item["image_path"],
                    question=item["question"],
                    model_answer=answer,
                    correct_answer=item["answer"],
                    is_correct=is_correct,
                )
            )

            if is_correct:
                correct_count += 1

        print(correct_count)
        print(len(samples))

        return cls(accuracy=correct_count / len(samples) if samples else 0, samples=samples)


    @classmethod
    def from_answers_extended(
        cls, 
        responses: List[str], 
        gt_dataset: List[dict[str, Any]],
        gt_questions_used: List[str] # Pass questions collected in benchmark function
    ) -> "VQABenchmarkResult":
        """
        Creates a benchmark result and computes all custom metrics (MAE/RMSE)
        by calling compute_metrics_test.
        """
        
        # 1. Prepare raw answers and questions
        gt_answers = [item["answer"] for item in gt_dataset]
        
        # 2. Compute all structured metrics (using the external function)
        # We rely on the external compute_metrics_test to do the heavy lifting
        # (You would define this function separately, as shown in the previous answer)
        all_metrics = compute_metrics_test(responses, gt_answers, gt_questions_used)
        
        # 3. Populate samples list (for debugging and review)
        samples = []
        
        # Note: You can no longer reliably calculate 'is_correct' for the mixed 
        # regression/classification task in one loop; we default to False 
        # or use the classification accuracy result.
        
        for i in range(len(responses)):
            samples.append(
                cls.Sample(
                    image_path=gt_dataset[i].get("image_path", ""),
                    question=gt_questions_used[i],
                    model_answer=responses[i],
                    correct_answer=gt_answers[i],
                    is_correct=False # Set to False or only True if it's a classification match
                )
            )

        # 4. Return the new result object
        return cls(
            accuracy=all_metrics.pop("classification_accuracy", 0.0),
            regression_metrics=all_metrics, # All MAE/RMSE scores go here
            samples=samples
        )


def benchmark(model, dataset: VQADataset, max_samples = None) -> VQABenchmarkResult:
    """
    Benchmark a VLM model on a dataset.

    Args:
        model: VLM model to evaluate
        dataset: Dataset to evaluate on
        max_samples: Maximum number of samples to evaluate

    Returns:
        Benchmark result
    """

    if len(dataset) == 0 or max_samples == 0:
        raise ValueError("Dataset or model is empty")

    # Limit the number of samples if specified
    if max_samples is not None:
        dataset_size = min(len(dataset), max_samples)
    else:
        dataset_size = len(dataset)

    import random

    sample_indices = random.sample(range(len(dataset)), dataset_size)

    # Extract questions and image paths
    questions = [dataset[i]["question"] for i in sample_indices]
    image_paths = [dataset[i]["image_path"] for i in sample_indices]
    answers = [dataset[i]["answer"] for i in sample_indices]
    # Get model answers
    responses = []
    gt_dataset = []
    mini_batch_size = 32
    import tqdm

    for i in tqdm.tqdm(range(0, dataset_size, mini_batch_size)):  # Process in batches
        batch_size = min(mini_batch_size, dataset_size - i)
        batch_questions = questions[i : i + batch_size]
        batch_image_paths = image_paths[i : i + batch_size]
        batch_indices = sample_indices[i : i + batch_size]

        batch_responses = model.answer(batch_image_paths, batch_questions)
        responses.extend(batch_responses)
        gt_dataset.extend([dataset[i] for i in batch_indices])
        print(f"\tProcessed {i + batch_size} samples")
        print(f"\tQuestions: {batch_questions}")
        print(f"\tResponses: {batch_responses}")
        print(f"\tAnswers: {answers[i : i + batch_size]}")

    return VQABenchmarkResult.from_answers(responses, gt_dataset, max_samples)


def compute_metrics_test(responses: list[str], gt_answers: list[str], gt_questions: list[str]) -> dict[str, float]:
    """Calculates all classification and regression metrics from decoded strings."""
    # Define a similarity threshold for a 'correct' fuzzy match (e.g., 90 out of 100)
    FUZZY_THRESHOLD = 90

    classification_counts = {
        'total_correct_strict': 0,
        'total_count': 0,
        'total_correct_fuzzy': 0,  # <-- NEW: Track fuzzy matches
        
        'dish_name_count': 0,
        'dish_name_correct_strict': 0,
        'dish_name_correct_fuzzy': 0, # <-- NEW

        'category_count': 0,
        'category_correct_strict': 0,
        'category_correct_fuzzy': 0, # NEW
        
        'cooking_method_count': 0,
        'cooking_method_correct_strict': 0,
        'cooking_method_correct_fuzzy': 0, # NEW
    }
    
    true_regression_values = [[] for _ in range(NUM_VALUES)]
    pred_regression_values = [[] for _ in range(NUM_VALUES)]
    true_class_labels = []
    pred_class_labels = []
    
    # --- Parsing Loop ---
    for sample_index, (pred_str, label_str) in enumerate(zip(responses, gt_answers)):
        
        clean_label = label_str.strip().lower()
        clean_pred = pred_str.strip().lower()

        # Check if it's a regression output (contains a comma in the clean label)
        if ',' in clean_label:
            # --- REGRESSION TASK LOGIC ---
            
            # 1. Parse GROUND TRUTH 
            try:
                true_list = [float(x.strip()) for x in clean_label.split(',')]
            except Exception:
                continue
            
            # 2. Parse PREDICTION (Handle parsing failure by defaulting to zero)
            try:
                pred_list = [float(x.strip()) for x in clean_pred.split(',')]
                if len(pred_list) != len(true_list):
                    pred_list = [0.0] * len(true_list) # Default to max error if structure is wrong
            except Exception:
                pred_list = [0.0] * len(true_list)
            
            # 3. Identify Task (Using the question, which is safer than the answer string itself)
            question = gt_questions[sample_index]
            q_lower = question.lower()
            
            if "calories" in q_lower or "fat" in q_lower:
                base_index = 0
            elif "total weight" in q_lower or "iron" in q_lower:
                base_index = 4
            else:
                continue 

            # Append values to the correct master lists
            for i in range(len(true_list)):
                true_regression_values[base_index + i].append(true_list[i])
                pred_regression_values[base_index + i].append(pred_list[i])
            
        else:
            # --- CLASSIFICATION TASK LOGIC ---
            clean_label = label_str.strip().lower()
            clean_pred = pred_str.strip().lower()
            question = gt_questions[sample_index] # Access the question
            
            true_class_labels.append(clean_label) # Keep tracking total samples
            
            is_correct = (clean_pred == clean_label)
    
            # --- NEW: Calculate Fuzzy Match Score ---
            # Use token_sort_ratio to be lenient on spacing, punctuation, and order
            similarity_score = fuzz.token_sort_ratio(clean_pred, clean_label)
            is_fuzzy_correct = (similarity_score >= FUZZY_THRESHOLD)
            is_strict_correct = (clean_pred == clean_label) # Simple matching is now strict

            # 1. Determine Question Type
            if "general name" in question.lower():
                q_type = 'dish_name'
            elif "category of food" in question.lower():
                q_type = 'category'
            elif "cooking method" in question.lower():
                q_type = 'cooking_method'
            else:
                q_type = None # Catch any uncategorized question
            
            # 2. Update Counters
            if q_type:
                classification_counts['total_count'] += 1
                classification_counts[f'{q_type}_count'] += 1
                
                if is_strict_correct:
                    classification_counts['total_correct_strict'] += 1
                    classification_counts[f'{q_type}_correct_strict'] += 1
                
                # --- NEW: Update Fuzzy Counters ---
                if is_fuzzy_correct:
                    classification_counts['total_correct_fuzzy'] += 1
                    classification_counts[f'{q_type}_correct_fuzzy'] += 1
            
            # Check for accuracy (using full cleaned string match)
            if clean_pred == clean_label: 
                pred_class_labels.append(clean_label)
            else:
                pred_class_labels.append(clean_pred)


    # --- Calculation Loop ---
    metrics = {}
    
    # A. Regression Metrics (MAE/RMSE/MAPE per value)
    for i, label_name in enumerate(ALL_LABELS):
        true_values = np.array(true_regression_values[i])
        pred_values = np.array(pred_regression_values[i])
        
        if len(true_values) > 0:
            # Calculate MAE and RMSE
            mae = mean_absolute_error(true_values, pred_values)
            rmse = np.sqrt(mean_squared_error(true_values, pred_values))
            
            # --- START MAPE CALCULATION ---
            
            # 1. Define safe denominator: use 1e-8 instead of 0 to prevent division by zero
            true_values_safe = np.where(true_values == 0, 1e-8, true_values) 
            
            # 2. Calculate the Absolute Percentage Error for every sample
            # |y_i - y_hat_i| / y_i
            absolute_percentage_error = np.abs((true_values - pred_values) / true_values_safe)
            
            # 3. Calculate MAPE (Mean of the Absolute Percentage Error), multiplied by 100 to get a percentage
            mape = np.mean(absolute_percentage_error) * 100
            
            # --- END MAPE CALCULATION ---
            
            metrics[f"{label_name}_mae"] = mae
            metrics[f"{label_name}_rmse"] = rmse
            metrics[f"{label_name}_mape"] = mape # <-- New Metric Added

    # B. Classification Metrics (Accuracy)
    if true_class_labels:
        correct_count = sum(1 for t, p in zip(true_class_labels, pred_class_labels) if t == p)
        metrics["classification_accuracy"] = correct_count / len(true_class_labels)
        
    # C. Classification Metrics (Total and Per-Type)
    if classification_counts['total_count'] > 0:
        total_count = classification_counts['total_count']
        
        # Total Accuracy (Strict and Fuzzy)
        metrics["total_accuracy_strict"] = classification_counts['total_correct_strict'] / total_count
        metrics["total_accuracy_fuzzy"] = classification_counts['total_correct_fuzzy'] / total_count # <-- NEW
        
        # Per-Type Accuracy (Strict and Fuzzy)
        for q_type in ['dish_name', 'category', 'cooking_method']:
            count = classification_counts[f'{q_type}_count']
            if count > 0:
                metrics[f"accuracy_{q_type}_strict"] = classification_counts[f'{q_type}_correct_strict'] / count
                metrics[f"accuracy_{q_type}_fuzzy"] = classification_counts[f'{q_type}_correct_fuzzy'] / count # <-- NEW
                
    return metrics


def benchmark_extended(model, dataset: VQADataset, mini_batch_size=32, max_samples=None) -> VQABenchmarkResult:
    """
    Benchmark a VLM model on a dataset using the extended metric calculation.
    """
    if len(dataset) == 0 or max_samples == 0:
        raise ValueError("Dataset or model is empty")

    if max_samples is not None:
        dataset_size = min(len(dataset), max_samples)
    else:
        dataset_size = len(dataset)

    import random

    sample_indices = random.sample(range(len(dataset)), dataset_size)

    # Extract ALL questions, image paths, and answers based on sampled indices
    questions = [dataset[i]["question"] for i in sample_indices]
    image_paths = [dataset[i]["image_path"] for i in sample_indices]
    answers = [dataset[i]["answer"] for i in sample_indices]
    
    # Get model answers
    responses = []
    
    # Initialize lists to be passed to from_answers_extended
    gt_dataset_items = []      # List of full dataset item dicts (was 'gt_dataset' below)
    gt_questions_used = []     # List of question strings used in the batch
    
    import tqdm

    for i in tqdm.tqdm(range(0, dataset_size, mini_batch_size)):
        batch_size = min(mini_batch_size, dataset_size - i)
        batch_questions = questions[i : i + batch_size]
        batch_image_paths = image_paths[i : i + batch_size]
        batch_indices = sample_indices[i : i + batch_size]

        batch_responses = model.answer(batch_image_paths, batch_questions)
        responses.extend(batch_responses)
        
        # --- FIX 1: Store the full dataset items ---
        batch_gt_items = [dataset[idx] for idx in batch_indices]
        gt_dataset_items.extend(batch_gt_items) 
        
        # --- FIX 2: Store the questions used for the full list ---
        gt_questions_used.extend(batch_questions)
        
        print(f"\tProcessed {i + batch_size} samples")
        print(f"\tQuestions: {batch_questions}")
        print(f"\tResponses: {batch_responses}")
        print(f"\tAnswers: {answers[i : i + batch_size]}")

    # --- FIX 3: Pass the correctly populated lists ---
    return VQABenchmarkResult.from_answers_extended(
        responses=responses, 
        gt_dataset=gt_dataset_items, 
        gt_questions_used=gt_questions_used
    )


if __name__ == "__main__":
    # Test the dataset
    dataset = VQADataset("train")
    print(f"Dataset size: {len(dataset)}")

    # Print a sample
    sample = dataset[0]
    print("\nSample:")
    print(f"Image: {sample['image_path']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
