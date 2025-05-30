import ast
import re
import unicodedata
from fractions import Fraction
import json
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss
from tqdm import tqdm
import os
from approximate_randomization import chanceByChanceDataFrame
from ner.predict import NERPredictor
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class MultiLabel:
    def __init__(self, config):
        self.config = config
        self.ner_predictor = NERPredictor()
        self.special_tokens = {
            'additional_special_tokens': [
                '[B-COLOR]',
                '[B-FOOD]',
                '[B-QUANTITY]',
                '[B-UNIT]',
                '[I-FOOD]',
                '[I-QUANTITY]',
                '[O]',
                '[ING_END]',
            ]
        }
        self.raw_dataset = config["dataset_path"] + 'diet_type_recipes.csv'
        self.dataset_with_ner = config["dataset_path"] + 'diet_type_recipes_with_ner.csv'
        self.filtered_list = [
            'healthy',
            'vegan',
            'low-carb',
            'gluten-free',
            'high-protein',
            'low-sugar'
        ]
        if not os.path.exists(self.dataset_with_ner):
            print("Dataset not found. We will create it.")
            self.preprocessing()
        else:
            print("Dataset found. We will use it.")

    @staticmethod
    def preprocessing_text(item):
        item = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', item)
        item = re.sub(r'\([^)]*\)', '', item)
        # use unicodedata
        item = unicodedata.normalize("NFKD", item)
        item = item.replace("⁄", "/")
        # if two number on middle have x between them multiply them and replace with new number
        item = re.sub(r'(\d+) (x) (\d+)', lambda x: str(int(x.group(1)) * int(x.group(3))), item)
        # convert fraction to decimal
        item = re.sub(r'(\d+/\d+)', lambda x: str(float(Fraction(x.group(1)))), item)
        # get after comma 4 digit
        item = re.sub(r'(\d+),(\d{4})', r'\1.\2', item)

        # Add unit mapping
        unit_mapping = {
            'tbsp': 'tablespoons',
            'tsp': 'teaspoons',
            'lb': 'pounds',
            'oz': 'ounces',
            'g': 'grams',
            'kg': 'kilograms',
            'ml': 'milliliters',
            'l': 'liters',
            'cup': 'cups',
            'pt': 'pints',
            'qt': 'quarts',
            'gal': 'gallons',
            'fl oz': 'fluid ounces',
            'pint': 'pints',
            'quart': 'quarts'
        }

        for full_unit, abbrev in unit_mapping.items():
            item = re.sub(rf'\b{full_unit}\b', abbrev, item)

        # Clean up spaces
        item = re.sub(r'\s+', ' ', item)
        item = re.sub(r' , ', ', ', item)

        # Clean up extra spaces
        item = re.sub(r'\s+', ' ', item)
        # Clean up spaces around commas
        item = re.sub(r'\s*,\s*', ', ', item)
        # Remove leading/trailing commas
        item = re.sub(r'^,\s*|\s*,$', '', item)

        return item.strip()

    def ner_predict(self, text_list: list):
        predictions = []
        for text in text_list:
            word_predictions = self.ner_predictor.predict_entities(text)
            predictions.append(word_predictions)
        return predictions

    def format_text_with_ner_tags(self, ner_tag_list):
        ner_results = []
        for prediction in ner_tag_list:
            output = []
            for word, tag in prediction:
                output.append(f"[{tag}]{word}")
            output.append("[ING_END]")
            ner_results.append(' '.join(output))
        return ner_results

    def preprocessing(self):
        df = pd.read_csv(self.raw_dataset)
        df["diets"] = df["diets"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["diets"] = df["diets"].apply(lambda x: [item.lower() for item in x])
        df["diets"] = df["diets"].apply(lambda x: [item for item in x if item in self.filtered_list])

        # Process ingredients
        df["ingredients"] = df["ingredients"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["processed_ingredients"] = df["ingredients"].apply(lambda x: [self.preprocessing_text(item) for item in x])

        # Create new field for NER results
        df["ner_results"] = df["ingredients"].apply(lambda x: x.copy())

        # NER prediction
        flat_list = []
        inds = []
        ingredient_indices = []
        for index, sublist in enumerate(df["processed_ingredients"].tolist()):
            for i, item in enumerate(sublist):
                flat_list.append(item)
                inds.append(index)
                ingredient_indices.append(i)
        ner_predictions = self.ner_predict(flat_list)
        ner_results = self.format_text_with_ner_tags(ner_predictions)

        # Update the new ner_results field instead of ingredients
        for index, (recipe_idx, ing_idx, ner_result) in enumerate(zip(inds, ingredient_indices, ner_results)):
            df.at[recipe_idx, "ner_results"][ing_idx] = ner_result

        df['ner_results'] = df['ner_results'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df = df[['ingredients', 'processed_ingredients', 'ner_results', 'diets']]
        df.to_csv(self.dataset_with_ner, index=False)

    def train_model(self):
        final_reports = []
        comparison_data = []  # List to store metrics for comparison
        
        df = pd.read_csv(self.dataset_with_ner)
        df["diets"] = df["diets"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["diets"] = df["diets"].apply(lambda x: [item.lower() for item in x])
        df["diets"] = df["diets"].apply(lambda x: [item for item in x if item in self.filtered_list])

        ner_flags = [True, False]
        for ner_flag in ner_flags:
            if ner_flag:
                print("Starting with NER results..")
                df = df[["ner_results", "diets"]]
                df["processed_ingredients"] = df["ner_results"].apply(lambda x: x if isinstance(x, str) else " ".join(x))
            else:
                print("Starting without NER results..")
                df = df[["processed_ingredients", "diets"]]
                df["processed_ingredients"] = df["processed_ingredients"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                df["processed_ingredients"] = df["processed_ingredients"].apply(lambda x: " ".join(x))

            # Create MultiLabelBinarizer for diet labels
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(df["diets"])
            
            # Split the data
            X_train, X_temp, y_train, y_temp = train_test_split(
                df["processed_ingredients"].values, y,
                test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=0.5, random_state=42
            )

            # Create custom dataset
            class DietDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, max_length=512):
                    self.texts = texts
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_length = max_length

                def __len__(self):
                    return len(self.texts)

                def __getitem__(self, idx):
                    text = str(self.texts[idx])
                    encoding = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze(),
                        'labels': torch.FloatTensor(self.labels[idx])
                    }

            # Initialize tokenizer and model
            for model_name in self.config["model_names"]:
                print(f"\n{'='*50}")
                print(f"Training {model_name}")
                print(f"{'='*50}")
                
                model_report = {
                    "ner_flag": ner_flag,
                    "model_name": model_name,
                    "epoch_results": []
                }

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(self.filtered_list),
                    problem_type="multi_label_classification"
                )

                # Add special tokens
                tokenizer.add_special_tokens(self.special_tokens)
                model.resize_token_embeddings(len(tokenizer))
                print("Tokenizer special tokens:", tokenizer.special_tokens_map)

                # Create datasets
                train_dataset = DietDataset(X_train, y_train, tokenizer)
                val_dataset = DietDataset(X_val, y_val, tokenizer)
                test_dataset = DietDataset(X_test, y_test, tokenizer)

                # Create dataloaders
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config["batch_size"],
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config["batch_size"],
                    shuffle=False
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.config["batch_size"],
                    shuffle=False
                )

                # Setup training
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("Using device:", device)
                model.to(device)
                optimizer = AdamW(model.parameters(), lr=2e-5)
                criterion = BCEWithLogitsLoss()

                # Training loop
                best_f1 = 0
                early_stopping = EarlyStopping(patience=3, min_delta=0.001)
                
                for epoch in range(self.config["num_epochs"]):
                    # Training phase
                    model.train()
                    total_loss = 0
                    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [Train]')
                    for batch in train_pbar:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        optimizer.zero_grad()
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        loss = criterion(outputs.logits, labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                        # Update progress bar with current loss
                        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                    avg_train_loss = total_loss / len(train_loader)
                    print(f"\nEpoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

                    # Validation phase
                    model.eval()
                    val_loss = 0
                    val_preds = []
                    val_labels = []
                    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [Val]')
                    with torch.no_grad():
                        for batch in val_pbar:
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['labels'].to(device)

                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            loss = criterion(outputs.logits, labels)
                            val_loss += loss.item()

                            preds = torch.sigmoid(outputs.logits) > 0.5
                            val_preds.extend(preds.cpu().numpy())
                            val_labels.extend(labels.cpu().numpy())

                    avg_val_loss = val_loss / len(val_loader)
                    print(f"Epoch {epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")

                    # Early stopping check
                    early_stopping(avg_val_loss)
                    if early_stopping.early_stop:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        break

                    # Calculate validation metrics
                    val_f1 = f1_score(val_labels, val_preds, average='samples')
                    val_precision = precision_score(val_labels, val_preds, average='samples')
                    val_recall = recall_score(val_labels, val_preds, average='samples')

                    val_accuracy = accuracy_score(val_labels, val_preds)
                    val_h_loss = hamming_loss(val_labels, val_preds)

                    # Store epoch results
                    epoch_result = {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "val_f1": val_f1,
                        "val_precision": val_precision,
                        "val_recall": val_recall,
                        "val_accuracy": val_accuracy,
                        "val_h_loss": val_h_loss,
                        "val_classification_report": classification_report(val_labels,
                                                                           val_preds,
                                                                           target_names=self.filtered_list,
                                                                           output_dict=True)
                    }
                    model_report["epoch_results"].append(epoch_result)

                    # Save best model based on validation F1
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        print(f"New best model saved! Validation F1 Score: {val_f1:.4f}")
                        model_save_path = f"{project_path}/best_model_{model_name.replace('/', '_')}_{'with_ner' if ner_flag else 'without_ner'}.pt"
                        torch.save(model.state_dict(), model_save_path)
                        model_report["best_model_path"] = model_save_path
                        model_report["best_val_f1"] = val_f1

                # Final evaluation on test set
                print("\nFinal Test Set Evaluation:")
                model.eval()
                test_preds = []
                test_labels = []
                test_pbar = tqdm(test_loader, desc='Final Test Evaluation')
                with torch.no_grad():
                    for batch in test_pbar:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        preds = torch.sigmoid(outputs.logits) > 0.5
                        test_preds.extend(preds.cpu().numpy())
                        test_labels.extend(labels.cpu().numpy())

                # Calculate test metrics
                test_f1 = f1_score(test_labels, test_preds, average='samples')
                test_precision = precision_score(test_labels, test_preds, average='samples')
                test_recall = recall_score(test_labels, test_preds, average='samples')
                test_accuracy = accuracy_score(test_labels, test_preds)
                test_h_loss = hamming_loss(test_labels, test_preds)
                # Store final test results
                model_report["test_results"] = {
                    "test_f1": test_f1,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_accuracy": test_accuracy,
                    "test_h_loss": test_h_loss,
                    "test_classification_report": classification_report(test_labels,
                                                                        test_preds,
                                                                        target_names=self.filtered_list,
                                                                        output_dict=True)
                }

                # After final test evaluation, store metrics for comparison
                comparison_data.append({
                    'approach': 'with_ner' if ner_flag else 'without_ner',
                    'model': model_name,
                    'f1_score': float(test_f1),
                    'precision': float(test_precision),
                    'recall': float(test_recall),
                    'accuracy': float(test_accuracy),
                    'hamming_loss': float(test_h_loss)
                })

                # Add per-class metrics for comparison
                per_class_metrics = {}
                # Convert lists to numpy arrays
                test_labels_array = np.array(test_labels)
                test_preds_array = np.array(test_preds)
                
                for i, label in enumerate(self.filtered_list):
                    label_f1 = f1_score(test_labels_array[:, i], test_preds_array[:, i])
                    label_precision = precision_score(test_labels_array[:, i], test_preds_array[:, i])
                    label_recall = recall_score(test_labels_array[:, i], test_preds_array[:, i])
                    
                    per_class_metrics[label] = {
                        'f1_score': float(label_f1),
                        'precision': float(label_precision),
                        'recall': float(label_recall)
                    }
                
                comparison_data[-1]['per_class_metrics'] = per_class_metrics

                # Add model report to final reports
                final_reports.append(model_report)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Perform statistical comparison for each model and metric
        metrics = ['f1_score', 'precision', 'recall', 'accuracy', 'hamming_loss']
        comparison_results = {}
        
        print("\nStatistical Comparison Results:")
        print(f"{'='*50}")
        
        for model_name in self.config["model_names"]:
            print(f"\nModel: {model_name}")
            print(f"{'='*30}")
            
            model_df = comparison_df[comparison_df['model'] == model_name]
            model_results = {}
            
            # Overall metrics comparison
            for metric in metrics:
                p_value = chanceByChanceDataFrame(
                    model_df,
                    split_column='approach',
                    compare_column=metric,
                    left_value='with_ner',
                    right_value='without_ner',
                    repetitions=1000
                )
                
                with_ner_value = float(model_df[model_df['approach'] == 'with_ner'][metric].values[0])
                without_ner_value = float(model_df[model_df['approach'] == 'without_ner'][metric].values[0])
                diff = with_ner_value - without_ner_value
                
                model_results[metric] = {
                    'p_value': float(p_value),
                    'is_significant': bool(p_value < 0.05),
                    'with_ner': with_ner_value,
                    'without_ner': without_ner_value,
                    'difference': float(diff)
                }
                
                print(f"\n{metric}:")
                print(f"P-value: {p_value:.4f}")
                if p_value < 0.05:
                    print(f"  - Statistically significant difference in {metric}")
                    print(f"  - Difference: {diff:.4f} ({'+' if diff > 0 else ''}{diff:.4f})")
                    print(f"  - With NER: {with_ner_value:.4f}")
                    print(f"  - Without NER: {without_ner_value:.4f}")
                else:
                    print(f"  - No statistically significant difference in {metric}")
            
            # Per-class comparison
            print("\nPer-Class Comparison:")
            print(f"{'='*30}")
            
            per_class_results = {}
            for label in self.filtered_list:
                print(f"\nLabel: {label}")
                class_results = {}
                
                for metric in ['f1_score', 'precision', 'recall']:
                    with_ner_value = model_df[model_df['approach'] == 'with_ner']['per_class_metrics'].iloc[0][label][metric]
                    without_ner_value = model_df[model_df['approach'] == 'without_ner']['per_class_metrics'].iloc[0][label][metric]
                    diff = with_ner_value - without_ner_value
                    
                    class_results[metric] = {
                        'with_ner': with_ner_value,
                        'without_ner': without_ner_value,
                        'difference': diff
                    }
                    
                    print(f"{metric}:")
                    print(f"  - With NER: {with_ner_value:.4f}")
                    print(f"  - Without NER: {without_ner_value:.4f}")
                    print(f"  - Difference: {diff:.4f} ({'+' if diff > 0 else ''}{diff:.4f})")
                
                per_class_results[label] = class_results
            
            model_results['per_class_comparison'] = per_class_results
            comparison_results[model_name] = model_results

        # Save all results to a JSON file
        results_file = f"{project_path}/training_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_reports': final_reports,
                'statistical_comparison': comparison_results
            }, f, indent=4)
        print(f"\nAll results saved to {results_file}")

        return model, mlb


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    print("Project path:", project_path)
    print("CUDA available:", torch.cuda.is_available())

    config = {
        "batch_size": 16,
        "num_epochs": 6,
        "dataset_path": project_path + "/datasets/",
        "model_names": [
            "FacebookAI/roberta-base",
            "distilbert/distilbert-base-uncased",
            "bert-base-uncased"
        ],
    }
    ml = MultiLabel(config=config)
    ml.train_model()

