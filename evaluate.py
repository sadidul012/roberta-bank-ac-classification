from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dataset import StringDataset
from tqdm import tqdm
from cutils import *
from datasets import load_metric
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def metric(y_tru, y_hat):
    cm = confusion_matrix(y_true=y_tru, y_pred=y_hat, labels=range(num_classes))
    true_positives = np.diag(cm)
    false_positives = []
    for x in range(num_classes):
        false_positives.append(sum(cm[:, x]) - cm[x, x])

    pr = []
    for tp, fp in zip(true_positives, false_positives):
        p = (tp / (tp + fp))
        p = p if pd.notnull(p) else 1
        pr.append((1/num_classes) * p)

    micro_acc = np.sum(true_positives) / (np.sum(false_positives) + np.sum(true_positives))
    macro_acc = np.sum(pr)
    print("\tmicro accuracy:", micro_acc)
    print("\tmacro accuracy:", macro_acc)

    return micro_acc, macro_acc


def test_roberta(model, tokenizer):
    accuracy = load_metric("accuracy")

    model.to(device)
    dataset = StringDataset(str(Path(data_location, "bank_account_classifier_test_set_processed.csv")), tokenizer)
    dataset.train_data = dataset.train_data
    progress_bar = tqdm(range(len(dataset.train_data)), leave=True, position=0)

    loss = []
    y_hat = []
    y_tru = []

    for batch in dataset.train_data:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
        loss.append(outputs.loss.item())

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        y_hat.extend(predictions.tolist())
        y_tru.extend(batch["labels"].tolist())
        accuracy.add_batch(predictions=predictions, references=batch["labels"])

        progress_bar.update(1)

    accuracy = accuracy.compute()
    # print(accuracy)
    # print("loss", np.mean(loss))
    micro, macro = metric(y_tru, y_hat)
    return accuracy["accuracy"], np.mean(loss), micro, macro


if __name__ == '__main__':
    # t = AutoTokenizer.from_pretrained(Path(data_location, "bank-classifier-roberta-fold-1"))
    # m = AutoModelForSequenceClassification.from_pretrained(Path(data_location, "bank-classifier-roberta-fold-1"), num_labels=num_classes)
    # test_roberta(m, t)

    results = []
    for i in range(10):
        print("Fold {}:".format(i+1))
        t = AutoTokenizer.from_pretrained(Path(data_location, "bank-classifier-roberta-fold-{}".format(i+1)))
        m = AutoModelForSequenceClassification.from_pretrained(Path(data_location, "bank-classifier-roberta-fold-{}".format(i+1)), num_labels=num_classes)
        a, l, mi, ma = test_roberta(m, t)

        results.append({"fold": i+1, "micro": mi*100, "macro": ma * 100, "loss": l, "accuracy": a * 100})
        break

    df = pd.DataFrame(results)
    df['micro'] = df['micro'].apply(lambda x: round(x, 2))
    df['macro'] = df['macro'].apply(lambda x: round(x, 2))
    df['accuracy'] = df['accuracy'].apply(lambda x: round(x, 2))

    df.sort_values(by=["macro", "accuracy"], inplace=True)

    df.to_csv(Path(data_location, "test_accuracy.csv"), index=False)

    print("\n\nbest folds - ")
    print(df[["fold", "macro", "accuracy", "loss"]])
