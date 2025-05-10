import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from simpletransformers.ner import NERModel, NERArgs
import matplotlib.pyplot as plt



df = pd.read_csv("helper/taste_iob.csv")

df = df[~df["labels"].isin(["B-DIET",
                            "I-DIET",
                            "B-EXAMPLE",
                            "I-EXAMPLE",
                            "B-EXCLUDED",
                            "I-EXCLUDED",
                            "B-POSSIBLE_SUBSTITUTE",
                            "I-POSSIBLE_SUBSTITUTE",
                            "B-TRADE_NAME",
                            "I-TRADE_NAME",
                            "B-PART",
                            "I-PART",
                            "B-PURPOSE",
                            "I-PURPOSE",
                            "B-TASTE",
                            "I-TASTE",
                            "B-EXCLUSIVE"])]



df = df.sample(frac=1).reset_index(drop=True)



train_data, eval_data = train_test_split(df, test_size=0.2, random_state=42)
if __name__ == '__main__':
    all_train_results = []
    all_eval_results = []
    # Configure the model
    model_args = NERArgs()
    model_args.train_batch_size = 16
    model_args.eval_batch_size = 16
    model_args.adam_epsilon = 1e-5
    model_args.max_grad_norm = 1.0
    model_args.max_seq_length = 128
    model_args.evaluate_during_training = True
    model_args.evaluate_each_epoch = True
    model_args.classification_report = True
    model_args.overwrite_output_dir = True
    model_args.learning_rate = 2e-5
  #  model_args.gradient_accumulation_steps = 1
    model_args.num_train_epochs = 3
    model_args.weight_decay = 0.01
    model_args.config = {"dropout": 0.1}
    model_args.optimizer = "AdamW"
    model_args.wandb_project = "NER"


    labels = df["labels"].unique().tolist()

    model = NERModel(
        "bert",
        "bert-base-cased",
        args=model_args,
        use_cuda=torch.cuda.is_available(),
        labels=labels,
    )
    model_args.use_deepspeed = False

    # Train the model
    train_results = model.train_model(train_data, eval_data=eval_data)

    predictions, raw_outputs = model.predict(["Tartufo Pasta with garlic flavoured butter and olive oil, egg yolk, parmigiano and pasta water."])

    print(predictions)

    plt.figure()
    plt.plot(train_results[1]["train_loss"], label="train")
    plt.plot(train_results[1]["eval_loss"], label="eval")
    plt.legend()
    plt.show()
    print("Done")
    # save the model
    model.save_model("models/ner_model")
    print("Model saved")
