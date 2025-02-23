import spacy
import pandas as pd
import os
import json
import unicodedata
import re
from fractions import Fraction
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_md")


# Load the data
PROJECT_MAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv("{}/ner/taste_v2.csv".format(PROJECT_MAIN_ROOT))

def preprocessing_text(item):
    # use unicodedata

    item = unicodedata.normalize("NFKD", str(item))
    item = item.replace("⁄", "/")
    # if two number on middle have x between them multiply them and replace with new number
    item = re.sub(r'(\d+) (x) (\d+)', lambda x: str(int(x.group(1)) * int(x.group(3))), item)
    # convert fraction to decimal
    item = re.sub(r'(\d+/\d+)', lambda x: str(float(Fraction(x.group(1)))), item)
    # get after comma 4 digit
    item = re.sub(r'(\d+),(\d{4})', r'\1.\2', item)

    # 2 0.5 should be 2.5
    item = re.sub(r'(\d+) (\d+\.\d+)', lambda x: str(float(x.group(1)) + float(x.group(2))), item)

    return item


def parse_annotations(annotations_str):
    try:
        annotations = json.loads(annotations_str.replace("'", '"'))
        parsed_annotations = []
        for annotation in annotations:
            span = eval(annotation["span"])
            parsed_annotations.append({
                "span": span,
                "type": annotation["type"],
                "entity": annotation["entity"]
            })
        return parsed_annotations
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []


# Function to generate IOB tags
def get_iob(text, annotations):
    tokens = word_tokenize(text)
    iob_tags = ["O"] * len(tokens)
    token_offsets = []

    offset = 0
    for token in tokens:
        start = text.find(token, offset)
        end = start + len(token)
        token_offsets.append((start, end))
        offset = end

    for annotation in annotations:
        for span in annotation['span']:
            start, end = span
            for i, (token_start, token_end) in enumerate(token_offsets):
                if start <= token_start < end:
                    iob_tags[i] = f"B-{annotation['type']}" if token_start == start else f"I-{annotation['type']}"
                elif token_start < start < token_end:
                    iob_tags[i] = f"I-{annotation['type']}"

    return list(zip(tokens, iob_tags))


# Initialize an empty list to store the results
results = []

# Process each row in the dataset
for index, row in df.iterrows():
    text = row["ingredients"]
    annotations = parse_annotations(row["ingredients_entities"])
    iob_output = get_iob(text, annotations)

    # Add the results to the list
    for token, tag in iob_output:
        results.append({
            "sentence_id": index,
            "words": token,
            "labels": tag
        })

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
results_df.to_csv('taste_iob.csv', index=False)

# Print out the results to verify
for index, row in results_df.iterrows():
    print(f"sentence_id: {row['sentence_id']}, words: {row['words']}, labels: {row['labels']}")
