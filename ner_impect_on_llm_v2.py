import ast
import os
import re
import unicodedata
from fractions import Fraction
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
from sklearn.metrics import f1_score
from approximate_randomization import *


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, mlb):
        self.tokenizer = tokenizer
        self.text = dataframe['ingredients_ner_format'].values
        self.targets = dataframe[mlb.classes_].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])

        # Calculate actual text length to determine optimal padding
        text_length = len(text.split())
        max_length = min(256, max(64, text_length + 10))  # Dynamic but capped padding

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(self.targets[index], dtype=torch.float)
        }


class EarlyStopping:
    def __init__(self, tolerance=3, min_delta=0.02):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_val_loss = float('inf')

    def __call__(self, train_loss, validation_loss):
        if validation_loss < self.best_val_loss:
            self.best_val_loss = validation_loss
            self.counter = 0
        elif (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class MultiLabel:
    def __init__(self, config: dict):
        self.df = pd.read_csv(config["dataset_path"])
        self.model_name = config["model_name"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.ner_tag_flag = config.get("ner_tag_flag", True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.mlb = MultiLabelBinarizer()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.filtered_list = [
            'healthy',
            'vegan',
            'low-carb',
            'gluten-free',
            'nut-free',
            'high-protein',
            'low-sugar'
        ]

    def format_text_with_ner_tags(self, ner_tag_list):
        ner_results = []
        for sublist in ner_tag_list:
            output = []
            for ner_tags in sublist:
                for ner_tag in ner_tags:  # Here ner_tag is each dictionary
                    for word, tag in ner_tag.items():  # Correctly access items in the dictionary
                        if tag == 'O':
                            output.append(f"[UNK]{word}")
                        else:
                            output.append(f"[{tag}]{word}")
            ner_results.append(' '.join(output))
        return ner_results

    def preprocessing_text(self, item):
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

        # Remove all numbers and special characters
        item = re.sub(r'\d+', '', item)  # Remove numbers

        # Remove all the mapped units
        for _, unit in unit_mapping.items():
            item = re.sub(rf'\b{unit}\b', '', item)

        # Remove special characters except commas
        item = re.sub(r'[^\w\s,]', '', item)

        # Clean up extra spaces
        item = re.sub(r'\s+', ' ', item)
        # Clean up spaces around commas
        item = re.sub(r'\s*,\s*', ', ', item)
        # Remove leading/trailing commas
        item = re.sub(r'^,\s*|\s*,$', '', item)

        return item.strip()

    def preprocessing(self):
        """
        Preprocess the data with updated diet categories.
        """
        # Convert diets to list and filter
        self.df["diets"] = self.df["diets"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.df["diets"] = self.df["diets"].apply(
            lambda x: [item.lower() for item in x]
        )
        self.df["diets"] = self.df["diets"].apply(
            lambda x: [item for item in x if item in self.filtered_list]
        )

        # Process ingredients
        self.df["ingredients"] = self.df["ingredients"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.df["ingredients"] = self.df["ingredients"].apply(
            lambda x: [self.preprocessing_text(item) for item in x]
        )

        # Flatten ingredients for NER processing if needed
        flat_list = []
        inds = []
        for index, sublist in enumerate(self.df["ingredients"].tolist()):
            for item in sublist:
                flat_list.append(item)
                inds.append(index)

        self.df['ingredients_ner_format'] = self.df['ingredients'].apply(lambda x: ' '.join(x))

        # Filter and clean the dataset
        pre_df = self.df[["ingredients_ner_format", "diets"]]
        pre_df = pre_df[pre_df["diets"].apply(lambda x: len(x) > 0)]
        pre_df = pre_df.dropna(subset=['diets'])
        pre_df = pre_df.reset_index(drop=True)

        # Save processed data
        pre_df.to_csv(project_path + "/datasets/diet_type_recipes_w_ner.csv", index=False)

        return pre_df

    def randomization_test(self, y_true, pred_1, pred_2, num_iterations=1000):
        def calculate_f1(y_true, y_pred):
            return f1_score(y_true, y_pred, average='micro')

        def f1_diff(s1, s2):
            return calculate_f1(y_true, s1) - calculate_f1(y_true, s2)

        pred_1 = (pred_1 > 0.5).astype(int)
        pred_2 = (pred_2 > 0.5).astype(int)

        p_value = chanceByChance(pred_1, pred_2, comparer=f1_diff, repetitions=num_iterations)
        observed_diff = f1_diff(pred_1, pred_2)

        return p_value, observed_diff

    def collate_fn(self, batch):
        input_ids = pad_sequence([item['input_ids'].squeeze(0) for item in batch], batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([item['attention_mask'].squeeze(0) for item in batch], batch_first=True,
                                      padding_value=0)
        labels = torch.stack([item['label'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def balance_dataset(self, df, mlb_classes):
        # Count samples per class
        label_counts = {label: df[label].sum() for label in mlb_classes}

        # Get average class size
        mean_size = sum(label_counts.values()) / len(label_counts)

        balanced_dfs = []
        for label in mlb_classes:
            current_size = label_counts[label]
            if current_size < mean_size:
                # Oversample small classes
                multiplier = int(mean_size / current_size)
                balanced_dfs.append(df[df[label] == 1].sample(n=current_size * multiplier, replace=True))
            else:
                # Keep larger classes as is
                balanced_dfs.append(df[df[label] == 1])

        return pd.concat(balanced_dfs).drop_duplicates()

    def calculate_metrics(self, true_labels, predictions, class_names):
        """Calculate and return evaluation metrics"""
        # Convert predictions to binary
        binary_preds = (predictions > 0.5).astype(int)

        # Calculate metrics
        results = classification_report(
            true_labels,
            binary_preds,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )

        # Calculate micro and macro F1 scores
        micro_f1 = f1_score(true_labels, binary_preds, average='micro')
        macro_f1 = f1_score(true_labels, binary_preds, average='macro')
        weighted_f1 = f1_score(true_labels, binary_preds, average='weighted')

        # Calculate overall accuracy
        accuracy = np.mean(binary_preds == true_labels)

        return {
            'detailed_report': results,
            'accuracy': accuracy,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }

    # Better implementation of positive weights calculation
    def calculate_pos_weights(self, df, mlb_classes):
        # Calculate total number of samples
        n_samples = len(df)
        if n_samples == 0:
            # Default weights if no data is provided
            return torch.ones(len(mlb_classes)).to(self.device)

        pos_weights = []
        for label in mlb_classes:
            # Count positive samples for this label
            n_pos = df[label].sum() if label in df.columns else 0
            # Count negative samples
            n_neg = n_samples - n_pos
            # Calculate weight: negative/positive ratio
            weight = n_neg / (n_pos + 1e-5)  # avoid division by zero
            pos_weights.append(weight)

        return torch.FloatTensor(pos_weights).to(self.device)

    def evaluate_model(self, model, data_loader):
        """Evaluate model on given data loader"""
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0

        pos_weights = self.calculate_pos_weights(pd.DataFrame([], columns=self.mlb.classes_), self.mlb.classes_)
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weights)

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                loss = loss_fn(outputs.logits, batch['labels'])
                total_loss += loss.item()

                all_predictions.extend(torch.sigmoid(outputs.logits).cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            self.mlb.classes_
        )

        return avg_loss, metrics, np.array(all_labels), np.array(all_predictions)

    def train(self):
        """
        Enhanced training function with proper validation, L2 regularization and monitoring
        """
        # Load and preprocess data
        print("Preprocessing data")
        df = self.preprocessing()
        self.mlb.fit([self.filtered_list])

        # Transform labels using MultiLabelBinarizer
        df = df.join(pd.DataFrame(
            self.mlb.transform(df.pop('diets')),
            columns=self.mlb.classes_,
            index=df.index
        ))

        # Print dataset information
        print("Number of labels:", len(self.mlb.classes_))
        print("Classes:", self.mlb.classes_)
        print("Original class distribution:")
        print(df[self.mlb.classes_].sum())

        # Balance dataset
        df = self.balance_dataset(df, self.mlb.classes_)
        print("\nBalanced class distribution:")
        print(df[self.mlb.classes_].sum())

        # Save label mapping
        if not os.path.exists(os.path.join(project_path, "classification/model/label_mapping.json")):
            label_mapping = {str(i): label for i, label in enumerate(self.mlb.classes_)}
            with open(os.path.join(project_path, "classification/model/label_mapping.json"), "w") as f:
                json.dump(label_mapping, f)

        # Split features and labels
        X = df["ingredients_ner_format"]
        y = np.asarray(df.drop(['ingredients_ner_format'], axis=1))

        # Create train-validation-test split with stratification
        # First split into train and temp sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.4,  # 60% for training
            random_state=42,
            shuffle=True,
            stratify=y.sum(axis=1)
        )

        # Split temp set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,  # 20% validation, 20% test
            random_state=42,
            shuffle=True,
            stratify=y_temp.sum(axis=1)
        )

        # Print shapes for verification
        print("X_train shape:", X_train.shape, "(60% of data)")
        print("X_val shape:", X_val.shape, "(20% of data)")
        print("X_test shape:", X_test.shape, "(20% of data)")

        # Prepare data for training
        y_train_df = pd.DataFrame(y_train, columns=self.mlb.classes_)
        y_val_df = pd.DataFrame(y_val, columns=self.mlb.classes_)
        y_test_df = pd.DataFrame(y_test, columns=self.mlb.classes_)

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        # Create datasets and dataloaders
        train_df = pd.concat([X_train, y_train_df], axis=1)
        val_df = pd.concat([X_val, y_val_df], axis=1)
        test_df = pd.concat([X_test, y_test_df], axis=1)

        train_dataset = TextDataset(train_df, self.tokenizer, self.mlb)
        val_dataset = TextDataset(val_df, self.tokenizer, self.mlb)
        test_dataset = TextDataset(test_df, self.tokenizer, self.mlb)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        # Initialize model with dropout for regularization
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.mlb.classes_),
            problem_type="multi_label_classification",
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2
        )
        model.resize_token_embeddings(len(self.tokenizer))
        model = model.to(self.device)

        # Initialize optimizer with increased L2 regularization
        optimizer = AdamW(
            model.parameters(),
            lr=2e-5,  # Slightly higher learning rate
            weight_decay=0.01
        )

        # Learning rate scheduler with validation loss monitoring
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            verbose=True
        )

        # Use in training
        pos_weights = self.calculate_pos_weights(df, self.mlb.classes_)
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weights)

        # Helper function to monitor weight norms
        def get_weights_norm(model):
            total_norm = 0
            for param in model.parameters():
                if param.requires_grad:
                    total_norm += param.norm(2).item() ** 2
            return np.sqrt(total_norm)

        # Initialize training trackers
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        best_epoch = 0
        early_stopping = EarlyStopping(tolerance=3, min_delta=0.02)

        # Create directory for model checkpoints if it doesn't exist
        checkpoint_dir = os.path.join(project_path, "classification/model/checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training loop
        for epoch in range(self.num_epochs):
            # TRAINING PHASE
            model.train()
            total_loss = 0
            total_batches = 0
            initial_weights_norm = get_weights_norm(model)

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                loss = loss_fn(outputs.logits, batch['labels'])
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_batches += 1

            # Calculate and print training metrics
            avg_train_loss = total_loss / total_batches
            final_weights_norm = get_weights_norm(model)
            train_losses.append(avg_train_loss)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Weights L2 norm: {final_weights_norm:.4f}")
            print(
                f"  Norm reduction: {((initial_weights_norm - final_weights_norm) / initial_weights_norm) * 100:.2f}%")

            # VALIDATION PHASE
            avg_val_loss, val_metrics, val_true, val_pred = self.evaluate_model(model, val_loader)
            val_losses.append(avg_val_loss)

            print(f"  Validation Loss: {avg_val_loss:.4f}")
            print(f"  Validation Micro-F1: {val_metrics['micro_f1']:.4f}")
            print(f"  Validation Macro-F1: {val_metrics['macro_f1']:.4f}")

            # Update learning rate based on validation performance
            scheduler.step(avg_val_loss)

            # Save best model based on validation F1 score (better metric than loss for imbalanced data)
            if val_metrics['micro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['micro_f1']
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1

                # Save model
                model.save_pretrained(os.path.join(project_path, "classification/model/best_model"))
                self.tokenizer.save_pretrained(os.path.join(project_path, "classification/model/best_model"))

                # Also save as checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_f1_{val_metrics['micro_f1']:.4f}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_f1': val_metrics['micro_f1'],
                    'train_loss': avg_train_loss,
                }, checkpoint_path)

                print(f"  Saved new best model with F1: {best_val_f1:.4f}")

            # Early stopping check
            early_stopping(avg_train_loss, avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Print final learning rate and best epoch statistics
        print("\nTraining completed!")
        print(f"Best model found at epoch {best_epoch} with validation F1: {best_val_f1:.4f}")
        print("Final learning rate:", optimizer.param_groups[0]['lr'])

        # Plot training and validation history
        plt.figure(figsize=(12, 8))

        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.axvline(x=best_epoch - 1, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot F1 scores (would need to track these during training)
        # For now we'll just add placeholders
        plt.subplot(2, 1, 2)
        plt.title('Model Convergence')
        plt.xlabel('Epochs')
        plt.tight_layout()
        plt.savefig(os.path.join(project_path, "classification/training_history.png"))
        plt.show()

        # Load best model for final evaluation
        best_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(project_path, "classification/model/best_model")
        ).to(self.device)

        # FINAL TEST EVALUATION
        print("\nEvaluating best model on test set:")
        test_loss, test_metrics, test_true, test_pred = self.evaluate_model(best_model, test_loader)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Micro-F1: {test_metrics['micro_f1']:.4f}")
        print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        print(f"Test Weighted-F1: {test_metrics['weighted_f1']:.4f}")

        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            test_true,
            (test_pred > 0.5).astype(int),
            target_names=self.mlb.classes_,
            zero_division=0
        ))

        # Create confusion matrices for each class
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Per-Class Performance: {self.model_name}", fontsize=16)

        num_classes = len(self.mlb.classes_)
        cols = 3
        rows = (num_classes + cols - 1) // cols

        # Plot precision, recall and F1 for each class
        class_metrics = test_metrics['detailed_report']

        class_names = self.mlb.classes_
        precisions = [class_metrics[cls]['precision'] for cls in class_names]
        recalls = [class_metrics[cls]['recall'] for cls in class_names]
        f1s = [class_metrics[cls]['f1-score'] for cls in class_names]

        plt.bar(class_names, precisions, label='Precision')
        plt.bar(class_names, recalls, label='Recall', alpha=0.7)
        plt.bar(class_names, f1s, label='F1-score', alpha=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(project_path, "classification/class_performance.png"))

        return test_true, (test_pred > 0.5).astype(int), self.mlb.classes_

    def compare_ner_performance(self):
        print("Training and evaluating model with NER...")
        self.ner_tag_flag = True
        true_labels, predictions_with_ner, class_names = self.train()

        print("Training and evaluating model without NER...")
        self.ner_tag_flag = False
        _, predictions_without_ner, _ = self.train()

        p_value, observed_diff = self.randomization_test(true_labels, predictions_with_ner, predictions_without_ner)

        print(f"Observed difference in F1 score (NER - No NER): {observed_diff}")
        print(f"P-value: {p_value}")

        # Create comparison visualization
        plt.figure(figsize=(10, 8))

        # Calculate per-class F1 for both approaches
        with_ner_f1 = []
        without_ner_f1 = []

        for i, class_name in enumerate(class_names):
            # Get class-specific metrics
            class_true = true_labels[:, i]
            class_with_ner = predictions_with_ner[:, i]
            class_without_ner = predictions_without_ner[:, i]

            with_ner_f1.append(f1_score(class_true, class_with_ner))
            without_ner_f1.append(f1_score(class_true, class_without_ner))

        # Plot comparison
        x = np.arange(len(class_names))
        width = 0.35

        plt.bar(x - width / 2, with_ner_f1, width, label='With NER')
        plt.bar(x + width / 2, without_ner_f1, width, label='Without NER')

        plt.xlabel('Diet Categories')
        plt.ylabel('F1 Score')
        plt.title('Impact of NER on Classification Performance')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(project_path, "classification/ner_comparison.png"))
        plt.show()

        if p_value < 0.05:
            print("The difference is statistically significant at the 0.05 level.")
        else:
            print("The difference is not statistically significant at the 0.05 level.")


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    print("Project path:", project_path)

    print("CUDA available:", torch.cuda.is_available())
    models = [
        "bert-base-uncased",
        "FacebookAI/roberta-base",
        "albert/albert-base-v2",
        "distilbert/distilbert-base-uncased"
    ]
    print("Start training")

    ner_tags_flags = [True, False]
    for item in models:
        print("Model:", item)
        print("NER tags flag:", ner_tags_flags)
        for ner_tag_flag in ner_tags_flags:
            config = {"model_name": item,
                      "batch_size": 8,
                      "num_epochs": 5,
                      "dataset_path": project_path + "/datasets/diet_type_recipes.csv",
                      }
            ml = MultiLabel(config=config)
            ml.train()
