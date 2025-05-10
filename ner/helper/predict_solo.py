from simpletransformers.ner import NERModel
import os
import torch

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    model = NERModel("bert", "{}/outputs/best_model".format(project_path),
                     use_cuda=torch.cuda.is_available())

    predictions, _ = model.predict(['Preheat oven to 350 degrees F (175 degrees C). ',
                                    'Place browned chicken breasts in a 9x13 inch baking dish. Brush with teriyaki sauce, then spoon on salad dressing. Sprinkle with cheese, green onions and bacon bits.'])

    print(predictions)