import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
project_path = os.path.dirname(os.path.abspath(__file__))
print(f"Project path: {project_path}")
df = pd.read_csv(project_path+"/dataset/taste_iob.csv")

# Filter out unwanted labels
df = df[~df["labels"].isin([
    "B-DIET", "I-DIET",
    "B-EXAMPLE", "I-EXAMPLE",
    "B-EXCLUDED", "I-EXCLUDED",
    "B-POSSIBLE_SUBSTITUTE", 
    "I-POSSIBLE_SUBSTITUTE",
    "B-TRADE_NAME", "I-TRADE_NAME",
    "B-PART", "I-PART",
    "B-PURPOSE", "I-PURPOSE",
    "B-TASTE", "I-TASTE",
    "B-EXCLUSIVE",
    "B-PROCESS", "I-PROCESS",
    "I-PHYSICAL_QUALITY", "B-PHYSICAL_QUALITY",
    "I-PROCESS", "B-PROCESS",
    "I-COLOR", "I-UNIT"
])]

# Save the filtered labels
unique_labels = sorted(df["labels"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Save label mappings
label_mappings = {
    "label2id": label2id,
    "id2label": id2label,
    "labels": list(unique_labels)  #
}

with open(os.path.join(project_path, "models/label_mappings.json"), "w") as f:
    json.dump(label_mappings, f, indent=2)

print("Saved label mappings to models/label_mappings.json")
print("Available labels:", unique_labels)

# Group by sentence_id so each row is a sentence with lists of words and labels
grouped = df.groupby("sentence_id").agg({
    "words": list,
    "labels": list
}).reset_index()

# Print unique labels to debug
print("Unique labels after filtering:", df["labels"].unique())

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["words"],
        truncation=True,
        max_length=128,
        padding="max_length",
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Convert label to string if it's not already
                current_label = str(label[word_idx]) if isinstance(label, list) else str(label)
                label_ids.append(label2id[current_label])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Initialize KFold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store results for each fold
fold_results = defaultdict(list)
all_predictions = []
all_true_labels = []

# Perform k-fold cross validation
for fold, (train_idx, val_idx) in enumerate(kf.split(grouped), 1):
    print(f"\nTraining Fold {fold}/{n_splits}")
    
    # Split data
    train_data = grouped.iloc[train_idx]
    val_data = grouped.iloc[val_idx]
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Initialize model for this fold
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(df["labels"].unique())
    ).to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"models/ner_model_fold_{fold}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        no_cuda=not torch.cuda.is_available()
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    
    # Train the model
    train_results = trainer.train()
    fold_results['train_loss'].append(train_results.training_loss)
    
    # Evaluate
    eval_results = trainer.evaluate()
    fold_results['eval_loss'].append(eval_results['eval_loss'])
    
    # Get predictions
    predictions = trainer.predict(tokenized_val)
    preds = np.argmax(predictions.predictions, axis=-1)
    
    # Flatten predictions and labels
    flat_preds = []
    flat_labels = []
    for pred, label in zip(preds, predictions.label_ids):
        mask = label != -100
        flat_preds.extend(pred[mask])
        flat_labels.extend(label[mask])
    
    # Convert numeric labels back to string labels
    flat_preds = [id2label[pred] for pred in flat_preds]
    flat_labels = [id2label[label] for label in flat_labels]
    
    # Store predictions and true labels
    all_predictions.extend(flat_preds)
    all_true_labels.extend(flat_labels)
    
    # Print fold results
    print(f"\nFold {fold} Results:")
    print(classification_report(flat_labels, flat_preds))
    
    # Save the model
    trainer.save_model(f"models/ner_model_fold_{fold}")
    print(f"Model for fold {fold} saved")

# Print overall results
print("\nOverall Cross-Validation Results:")
print("\nAverage Training Loss:", np.mean(fold_results['train_loss']))
print("Average Evaluation Loss:", np.mean(fold_results['eval_loss']))
print("\nOverall Classification Report:")
print(classification_report(all_true_labels, all_predictions))

# Save overall results
with open('cross_validation_results.json', 'w') as f:
    results = {
        'average_training_loss': float(np.mean(fold_results['train_loss'])),
        'average_eval_loss': float(np.mean(fold_results['eval_loss'])),
        'classification_report': classification_report(all_true_labels, all_predictions, output_dict=True)
    }
    json.dump(results, f, indent=2)

# Plot training results
plt.figure(figsize=(12, 6))
plt.plot(fold_results['train_loss'], label='Training Loss')
plt.plot(fold_results['eval_loss'], label='Evaluation Loss')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Across Folds')
plt.legend()
plt.savefig('cross_validation_losses.png')
plt.close()
