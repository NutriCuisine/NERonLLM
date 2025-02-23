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

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class MultiLabel:
    def __init__(self, config: dict):
        self.df = pd.read_csv(config["dataset_path"])
        self.model_name = config["model_name"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]

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

    def train(self):
        """
        Enhanced training function with L2 regularization and comprehensive monitoring
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

        # Create train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            random_state=42,
            shuffle=True,
            stratify=y.sum(axis=1)
        )

        # Print shapes for verification
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)

        # Prepare data for training
        y_train_df = pd.DataFrame(y_train, columns=self.mlb.classes_)
        y_test_df = pd.DataFrame(y_test, columns=self.mlb.classes_)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        # Create datasets and dataloaders
        train_df = pd.concat([X_train, y_train_df], axis=1)
        test_df = pd.concat([X_test, y_test_df], axis=1)
        train_dataset = TextDataset(train_df, self.tokenizer, self.mlb)
        test_dataset = TextDataset(test_df, self.tokenizer, self.mlb)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
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

        # Learning rate scheduler and loss function
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            verbose=True
        )

        # Better implementation
        def calculate_pos_weights(df, mlb_classes):
            # Calculate total number of samples
            n_samples = len(df)

            pos_weights = []
            for label in mlb_classes:
                # Count positive samples for this label
                n_pos = df[label].sum()
                # Count negative samples
                n_neg = n_samples - n_pos
                # Calculate weight: negative/positive ratio
                weight = n_neg / (n_pos + 1e-5)  # avoid division by zero
                pos_weights.append(weight)

            return torch.FloatTensor(pos_weights).to(self.device)

        # Use in training
        pos_weights = calculate_pos_weights(df, self.mlb.classes_)
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
        early_stopping = EarlyStopping(tolerance=3, min_delta=0.02)

        # Training loop
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0
            total_batches = 0
            initial_weights_norm = get_weights_norm(model)

            # Training phase
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
            avg_loss = total_loss / total_batches
            final_weights_norm = get_weights_norm(model)
            train_losses.append(avg_loss)

            print(f"\nEpoch {epoch + 1}:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Weights L2 norm: {final_weights_norm:.4f}")
            print(
                f"  Norm reduction: {((initial_weights_norm - final_weights_norm) / initial_weights_norm) * 100:.2f}%")

            # Validation phase
            model.eval()
            total_val_loss = 0
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    loss = loss_fn(outputs.logits, batch['labels'])
                    total_val_loss += loss.item()

                    val_predictions.extend(torch.sigmoid(outputs.logits).cpu().numpy())
                    val_true_labels.extend(batch['labels'].cpu().numpy())

            # Calculate validation metrics
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"  Validation Loss: {avg_val_loss:.4f}")

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save_pretrained(project_path + "/classification/model/best_model")
                self.tokenizer.save_pretrained(project_path + "/classification/model/best_model")
                print("  Saved new best model")

            # Early stopping check
            early_stopping(avg_loss, avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Print final learning rate
        print("\nFinal learning rate:", optimizer.param_groups[0]['lr'])

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Final evaluation
        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                predictions.extend(torch.sigmoid(outputs.logits).cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        # Convert predictions to binary values and print results
        true_labels_array = np.array(true_labels)
        predictions_array = np.array(predictions)
        predictions = (predictions_array > 0.5).astype(int)

        print("\nModel name:", self.model_name)
        print("Overall accuracy:", np.mean(predictions == true_labels_array))
        print(classification_report(
            true_labels_array,
            predictions,
            target_names=self.mlb.classes_,
            zero_division=0
        ))

        return true_labels_array, predictions, self.mlb.classes_

    def compare_ner_performance(self):
        print("Training and evaluating model with NER...")
        self.ner_tag_flag = True
        true_labels, predictions_with_ner, _ = self.train()

        print("Training and evaluating model without NER...")
        self.ner_tag_flag = False
        _, predictions_without_ner, _ = self.train()

        p_value, observed_diff = self.randomization_test(true_labels, predictions_with_ner, predictions_without_ner)

        print(f"Observed difference in F1 score (NER - No NER): {observed_diff}")
        print(f"P-value: {p_value}")

        if p_value < 0.05:
            print("The difference is statistically significant at the 0.05 level.")
        else:
            print("The difference is not statistically significant at the 0.05 level.")


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    print("Project path:", project_path)
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
