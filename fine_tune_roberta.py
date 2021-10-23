from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler, logging
import torch
from dataset import StringDataset
from tqdm import tqdm
import numpy as np
from cutils import *
from datasets import load_metric
import pandas as pd
from time import process_time
from datetime import timedelta


logging.set_verbosity_error()


def fine_tune_roberta(train_data, eval_data, tokenizer, num_epochs=3, version=""):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = len(train_data)
    num_eval_steps = len(eval_data)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    losses = []
    accuracies = []
    try:
        for epoch in range(num_epochs):
            progress_bar = tqdm(range(num_training_steps + num_eval_steps), desc="Epoch{} {}/{}".format(version, epoch + 1, num_epochs), leave=True, position=0)

            model.train()
            for batch in train_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                losses.append(loss.item())
                loss.backward()

                # update weights and biases
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix({'avg_loss': np.average(losses[-1000:])})

            accuracy = load_metric("accuracy")

            for batch in eval_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                accuracy.add_batch(predictions=predictions, references=batch["labels"])

                progress_bar.update(1)

            accuracy = accuracy.compute()
            accuracies.append(accuracy["accuracy"])
            progress_bar.set_postfix({'avg_loss': np.average(losses[-num_training_steps:]), 'val_accuracy': accuracy["accuracy"]})
    except KeyboardInterrupt:
        pass

    model.save_pretrained(Path(data_location, "bank-classifier-roberta{}".format(version)))
    tokenizer.save_pretrained(Path(data_location, "bank-classifier-roberta{}".format(version)))

    return np.average(accuracies)


if __name__ == '__main__':
    start_time = process_time()
    tokenize = AutoTokenizer.from_pretrained('roberta-base')
    # dataset = StringDataset(str(Path(data_location, "bank_account_classifier_training_set_processed.csv")), tokenize, test_split=0.2, batch_size=16)
    # print(len(dataset.train_data), len(dataset.eval_data))
    # fine_tune_roberta(dataset.train_data, dataset.eval_data, tokenize)

    dataset = StringDataset(str(Path(data_location, "bank_account_classifier_training_set_processed.csv")), tokenize, ten_fold=True, batch_size=16)
    a = []
    for i, (train_fold, test_fold) in enumerate(zip(dataset.train_data, dataset.eval_data)):
        accuracy = fine_tune_roberta(train_fold, test_fold, tokenize, version="-fold-{}".format(i+1), num_epochs=3)
        a.append({"fold": i+1, "accuracy": accuracy})
        # print(len(train_fold), len(test_fold), i)

    a = pd.DataFrame(a)
    a.to_csv(Path(data_location, "accuracy.csv"), index=False)
    print("process time:", timedelta(seconds=process_time()-start_time))
