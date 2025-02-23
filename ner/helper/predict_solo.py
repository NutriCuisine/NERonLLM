from simpletransformers.ner import NERModel
import os
import torch
from typing import List, Dict
import re


project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



RECIPE_TOKENS = {
    'HIGH_PROTEIN': '<high-protein>',
    'LOW_CARB': '<low-carb>',
    'HEALTHY': '<healthy>',
    'VEGAN': '<vegan>'
}


def convert_ner_to_template(ner_predictions: List[List[Dict[str, str]]]) -> List[dict]:
    """
    Converts NER predictions to a template format, focusing only on FOOD, QUANTITY, and UNIT tags.
    """
    TARGET_LABELS = {'FOOD', 'QUANTITY', 'UNIT'}
    results = []

    for sentence_predictions in ner_predictions:
        # Reconstruct full text
        full_text = ' '.join(list(word.keys())[0] for word in sentence_predictions)

        # Initialize template components
        template_words = []
        target_words = []
        extra_id_counter = 0

        i = 0
        while i < len(sentence_predictions):
            word_dict = sentence_predictions[i]
            word = list(word_dict.keys())[0]
            label = list(word_dict.values())[0]
            base_label = label.split('-')[-1] if '-' in label else label

            # Only process if it's one of our target labels
            if base_label in TARGET_LABELS and (label.startswith('B-') or label.startswith('I-')):
                entity_words = [word]
                current_label = base_label

                # Look ahead for continuation of entity (I- tags)
                j = i + 1
                while j < len(sentence_predictions):
                    next_word_dict = sentence_predictions[j]
                    next_word = list(next_word_dict.keys())[0]
                    next_label = list(next_word_dict.values())[0]
                    next_base_label = next_label.split('-')[-1] if '-' in next_label else next_label

                    if next_label.startswith('I-') and next_base_label == current_label:
                        entity_words.append(next_word)
                        j += 1
                    else:
                        break

                # Add to templates
                template_words.append(f"<extra_id_{extra_id_counter}>")
                target_words.append(f"<extra_id_{extra_id_counter}> {' '.join(entity_words)}")
                extra_id_counter += 1
                i = j

            else:
                template_words.append(word)
                i += 1

        # Build final input and target texts
        input_text = f"{RECIPE_TOKENS['HIGH_PROTEIN']} {' '.join(template_words)}"
        target_text = ' '.join(target_words)

        results.append({
            'full_text': full_text,
            'input_text': input_text,
            'target_text': target_text
        })

    return results


if __name__ == '__main__':
    model = NERModel("bert", "{}/ner/outputs/best_model".format(project_path),
                     use_cuda=torch.cuda.is_available())

    predictions, _ = model.predict(['Preheat oven to 350 degrees F (175 degrees C). ',
                                    'Place browned chicken breasts in a 9x13 inch baking dish. Brush with teriyaki sauce, then spoon on salad dressing. Sprinkle with cheese, green onions and bacon bits.'])

    # Convert to template format
    templates = convert_ner_to_template(predictions)

    # Print results
    for template in templates:
        print(f"# full text: {template['full_text']}")
        print(f"# input_text = {template['input_text']}")
        print(f"# target_text = {template['target_text']}")
        print("\n")
