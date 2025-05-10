import ast
import os
import re
import unicodedata
from fractions import Fraction
import json
import matplotlib.pyplot as plt
import pandas as pd
import torch
from simpletransformers.ner import NERModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss
from tqdm import tqdm
import os


class MultiLabel:
    def __init__(self, config):
        self.config = config
        self.special_tokens = {
            'additional_special_tokens': [
                '[B-UNIT]',
                '[B-PROCESS]',
                '[O]',
                '[B-QUANTITY]',
                '[B-FOOD]',
                '[I-FOOD]',
                '[B-PHYSICAL_QUALITY]',
                '[I-QUANTITY]',
                '[I-PROCESS]',
                '[B-COLOR]',
                '[I-PHYSICAL_QUALITY]',
                '[I-UNIT]',
                '[I-COLOR]'
            ]
        }
        self.raw_dataset = config["dataset_path"] + 'diet_type_recipes.csv'
        self.dataset_with_ner = config["dataset_path"] + 'diet_type_recipes_with_ner.csv'
        self.filtered_list = [
            'healthy',
            'vegan',
            'low-carb',
            'gluten-free',
            'nut-free',
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

    def ner_predict(self, text_list:list):
        ner_model = NERModel(
            "bert",
            project_path + "/ner/outputs/best_model",
            use_cuda=torch.cuda.is_available()
        )
        # Predict NER tags
        predictions, _ = ner_model.predict(text_list)
        return predictions

    def format_text_with_ner_tags(self, ner_tag_list):
        ner_results = []
        for sublist in ner_tag_list:
            output = []
            for ner_tag in sublist:
                if isinstance(ner_tag, dict):
                    for word, tag in ner_tag.items():
                        if tag == 'O':
                            output.append(f"[UNK]{word}")
                        else:
                            output.append(f"[{tag}]{word}")
                else:
                    output.append(f"[UNK]{ner_tag}")
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
        df = pd.read_csv(self.dataset_with_ner)
        df["diets"] = df["diets"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["diets"] = df["diets"].apply(lambda x: [item.lower() for item in x])
        df["diets"] = df["diets"].apply(lambda x: [item for item in x if item in self.filtered_list])

        ner_flags = [True, False]
        for ner_flag in ner_flags:
            if ner_flag:
                print("Starting with NER results..")
                df = df[["ner_results", "diets"]]
                # For NER results, we don't need to use literal_eval since it's already a string
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
                print("Model name:", model_name)
                # Create new dictionary for each model and NER flag combination
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

                    # Calculate validation metrics
                    val_f1 = f1_score(val_labels, val_preds, average='weighted')
                    val_precision = precision_score(val_labels, val_preds, average='weighted')
                    val_recall = recall_score(val_labels, val_preds, average='weighted')
                    val_accuracy = accuracy_score(val_labels, val_preds)
                    val_h_loss = hamming_loss(val_labels, val_preds)

                    print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']} Results:")
                    print(f"Training Loss: {total_loss/len(train_loader):.4f}")
                    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
                    print("\nValidation Metrics:")
                    print(f"F1 Score: {val_f1:.4f}")
                    print(f"Precision: {val_precision:.4f}")
                    print(f"Recall: {val_recall:.4f}")
                    print(f"Accuracy: {val_accuracy:.4f}")
                    print(f"Hamming Loss: {val_h_loss:.4f}")
                    print("\nValidation Classification Report:")
                    print(classification_report(val_labels, val_preds, target_names=self.filtered_list))
                    print("-" * 50)

                    # Store epoch results
                    epoch_result = {
                        "epoch": epoch + 1,
                        "train_loss": total_loss / len(train_loader),
                        "val_loss": val_loss / len(val_loader),
                        "val_f1": val_f1,
                        "val_precision": val_precision,
                        "val_recall": val_recall,
                        "val_accuracy": val_accuracy,
                        "val_h_loss": val_h_loss,
                        "val_classification_report": classification_report(val_labels, val_preds, target_names=self.filtered_list)
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
                test_f1 = f1_score(test_labels, test_preds, average='weighted')
                test_precision = precision_score(test_labels, test_preds, average='weighted')
                test_recall = recall_score(test_labels, test_preds, average='weighted')
                test_accuracy = accuracy_score(test_labels, test_preds)
                test_h_loss = hamming_loss(test_labels, test_preds)

                print("\nTest Set Results:")
                print(f"F1 Score: {test_f1:.4f}")
                print(f"Precision: {test_precision:.4f}")
                print(f"Recall: {test_recall:.4f}")
                print(f"Accuracy: {test_accuracy:.4f}")
                print(f"Hamming Loss: {test_h_loss:.4f}")
                print("\nTest Set Classification Report:")
                print(classification_report(test_labels, test_preds, target_names=self.filtered_list))

                # Store final test results
                model_report["test_results"] = {
                    "test_f1": test_f1,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_accuracy": test_accuracy,
                    "test_h_loss": test_h_loss,
                    "test_classification_report": classification_report(test_labels, test_preds, target_names=self.filtered_list)
                }

                # Add model report to final reports
                final_reports.append(model_report)

        # Save all results to a JSON file
        results_file = f"{project_path}/training_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_reports, f, indent=4)
        print(f"\nAll results saved to {results_file}")

        return model, mlb


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    print("Project path:", project_path)
    print("CUDA available:", torch.cuda.is_available())

    config = {
        "batch_size": 8,
        "num_epochs": 4,
        "dataset_path": project_path + "/datasets/",
        "model_names": [
            "bert-base-uncased",
            "FacebookAI/roberta-base",
            "albert/albert-base-v2",
            "distilbert/distilbert-base-uncased"
        ],
    }
    ml = MultiLabel(config=config)
    ml.train_model()

