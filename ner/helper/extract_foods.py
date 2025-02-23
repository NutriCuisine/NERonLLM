from simpletransformers.ner import NERModel
import os
import torch
import pandas as pd
import ast
import string
import json
from itertools import chain
import unicodedata
import re


def unicode_fraction_to_float(string):
    def convert_fraction(match):
        numerator = ord(match.group(1)) - 0x2080
        denominator = ord(match.group(2)) - 0x2080
        return str(numerator / denominator)

    string = re.sub(r'(\u2189)', '0', string)  # Replace '0/3' fraction with '0'
    string = re.sub(r'[\u00BC-\u00BE\u2150-\u215E]', lambda x: str(unicodedata.numeric(x.group())), string)
    string = re.sub(r'(\u2070-\u2079)⁄(\u2070-\u2079)', convert_fraction, string)
    return string



project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def process_ingredients(data):
    processed_data = []

    for item in data:
        quantity = ""
        unit = ""
        food = ""
        quantity_found = False
        unit_found = False

        for entity in item:
            for key, value in entity.items():
                key = unicode_fraction_to_float(key)

                if (value.startswith('B-QUANTITY') or value.startswith('I-QUANTITY')) and not quantity_found:
                    quantity += key + " "
                    if value.startswith('B-QUANTITY'):
                        quantity_found = True
                elif (value.startswith('B-UNIT') or value.startswith('I-UNIT')) and not unit_found:
                    unit += key + " "
                    if value.startswith('B-UNIT'):
                        unit_found = True
                elif value.startswith('B-FOOD') or value.startswith('I-FOOD'):
                    food += key + " "

        quantity = quantity.strip()
        unit = unit.strip()
        food = food.strip()

        if 'x' in quantity:
            quantity = quantity.split('x')[-1].strip()

        food = re.sub(r'[,.]$', '', food)
        if not unit:
            unit = 'unit'

        if not quantity:
            quantity = '1'

        # Convert quantity to float
        try:
            quantity = float(quantity)
        except ValueError:
            # If conversion fails, try to evaluate the expression
            try:
                quantity = float(eval(quantity))
            except:
                # If evaluation fails, set quantity to None
                quantity = None

        # Only add the ingredient if quantity is not None
        if quantity is not None:
            processed_data.append({
                'Quantity': quantity,
                'Unit': unit,
                'Food': food
            })

    return processed_data


def remove_parentheses(text):
    return re.sub(r'\([^)]*\)', '', text).strip()


def unisplit(ingredient):
    ingredient_without_parentheses = remove_parentheses(ingredient)
    match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)(.*)', ingredient_without_parentheses)
    if match:
        quantity, unit, rest = match.groups()
        return f"{quantity} {unit} {rest.strip()}"
    return ingredient_without_parentheses


def process_recipe(ingredients, nutrition):
    def safe_get(dict, key, default="0"):
        value = dict.get(key, default)
        value = value.split()[0] if isinstance(value, str) else str(value)
        try:
            value = float(value)
        except ValueError:
             value = float(eval(value))
        return value

    recipe_data = {
        "ingredients": ingredients,
        "nutrition_per_serving": {
            "calories": safe_get(nutrition, 'calories'),
            "fat": {
                "total": safe_get(nutrition, 'fatContent'),
                "saturated": safe_get(nutrition, 'saturatedFatContent')
            },
            "carbohydrates": {
                "total": safe_get(nutrition, 'carbohydrateContent'),
                "sugar": safe_get(nutrition, 'sugarContent'),
                "fiber": safe_get(nutrition, 'fiberContent')
            },
            "protein": safe_get(nutrition, 'proteinContent'),
            "sodium": safe_get(nutrition, 'sodiumContent')
        },
        "nutrition_units": {
            "calories": "calories",
            "fat": "grams",
            "carbohydrates": "grams",
            "protein": "grams",
            "sodium": "milligrams"
        }
    }
    return recipe_data


if __name__ == '__main__':
    model = NERModel("bert", f"{project_path}/ner/outputs/best_model",
                     use_cuda=torch.cuda.is_available())

    df = pd.read_csv(f"{project_path}/datasets/diet_type_recipes.csv")[:10000]

    df['nutrition'] = df['nutrition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["ingredients"] = df["ingredients"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["ingredients"] = df["ingredients"].apply(lambda x: [unisplit(item) for item in x if unisplit(item) is not None])

    # Flatten the list of ingredients
    all_ingredients = list(chain.from_iterable(df["ingredients"]))

    # Predict for all ingredients at once
    predictions, _ = model.predict(all_ingredients)

    # Process all predictions
    processed_ingredients = process_ingredients(predictions)

    # Split processed ingredients back into recipes
    ingredient_counts = df["ingredients"].apply(len)
    split_indices = ingredient_counts.cumsum().tolist()
    split_indices.insert(0, 0)

    processed_recipes = []
    for i, (index, row) in enumerate(df.iterrows()):
        start = split_indices[i]
        end = split_indices[i + 1]
        recipe_ingredients = processed_ingredients[start:end]
        recipe_data = process_recipe(recipe_ingredients, row['nutrition'])
        processed_recipes.append(recipe_data)

    # Save the processed data
    with open(f"{project_path}/datasets/processed_recipes.json", 'w') as f:
        json.dump(processed_recipes, f, indent=2)

    print(f"Processed {len(processed_recipes)} recipes. Data saved to processed_recipes.json")