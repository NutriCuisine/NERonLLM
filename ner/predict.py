import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import os

class NERPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        project_path = os.path.dirname(os.path.abspath(__file__))

        # Load label mappings
        with open(os.path.join(project_path, "models/label_mappings.json")) as f:
            mappings = json.load(f)
        self.label2id = mappings["label2id"]
        self.id2label = {int(k): v for k, v in mappings["id2label"].items()}

        # Load model and tokenizer
        model_path = os.path.join(project_path, "models/ner_model_fold_1")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print("Model and tokenizer loaded successfully.")


    def predict_entities(self, text):
        words = text.strip().split()
        encoding = self.tokenizer(words, is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt")
        word_ids = encoding.word_ids()
        inputs = {k: v.to(self.device) for k, v in encoding.items() if k != "offset_mapping"}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].tolist()

        word_predictions = []
        seen = set()
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            label_id = predictions[idx]
            word_predictions.append((words[word_id], self.id2label[label_id]))
        return word_predictions

# Test
# test_sentences = [
#     "1 tsp oil",
#     "2 cups flour",
#     "1/2 cup olive oil",
#     "3 tablespoons sugar"
# ]
#
# for sentence in test_sentences:
#     print(f"\nSentence: {sentence}")
#     for word, label in predict_entities(sentence):
#         print(f" - {word}: {label}")
