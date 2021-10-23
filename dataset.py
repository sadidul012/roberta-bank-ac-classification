from datasets import load_dataset
from torch.utils.data import DataLoader
from cutils import data_location
from pathlib import Path
from transformers import AutoTokenizer


class StringDataset:
    """
    Dataset for fine tuning question classification model
    """
    def tokenize_function(self, examples):
        """
        Tokenize Input and output for training and testing
        :param examples: Question and sentences
        :return:
        """
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)

    def __init__(self, location, tokenizer, batch_size=8, test_split=None, ten_fold=False):
        """
        :param tokenizer: Bert Base Cased tokenizer
        :param batch_size: batch size for each batch
        """
        self.tokenizer = tokenizer
        # Load dataset csv file

        if ten_fold:
            val_split = load_dataset("csv", data_files=[location], split=[
                f'train[{k}%:{k + 10}%]' for k in range(0, 100, 10)
            ])

            dataset = load_dataset("csv", data_files=[location], split=[
                f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)
            ])

            for i in range(len(dataset)):
                dataset[i] = dataset[i].remove_columns(['DB_NAME', 'bankAccountBankName', 'bankAccountBranch', 'bankAmount', 'lettering_code', 'payment_day', 'payment_month', 'payment_year'])
                dataset[i] = dataset[i].rename_column("full_bank_description", "text")
                dataset[i] = dataset[i].map(self.tokenize_function, batched=True)
                dataset[i] = dataset[i].remove_columns(["text"])
                dataset[i].set_format("torch")
                dataset[i] = DataLoader(dataset[i], batch_size=batch_size)

                val_split[i] = val_split[i].remove_columns(['DB_NAME', 'bankAccountBankName', 'bankAccountBranch', 'bankAmount', 'lettering_code', 'payment_day', 'payment_month', 'payment_year'])
                val_split[i] = val_split[i].rename_column("full_bank_description", "text")
                val_split[i] = val_split[i].map(self.tokenize_function, batched=True)
                val_split[i] = val_split[i].remove_columns(["text"])
                val_split[i].set_format("torch")
                val_split[i] = DataLoader(val_split[i], batch_size=batch_size)

            self.train_data = dataset
            self.eval_data = val_split
        else:
            dataset = load_dataset("csv", data_files=[location])
            dataset = dataset.remove_columns(['DB_NAME', 'bankAccountBankName', 'bankAccountBranch', 'bankAmount', 'lettering_code', 'payment_day', 'payment_month', 'payment_year'])
            dataset = dataset.rename_column("full_bank_description", "text")

            if test_split:
                dataset = dataset["train"].train_test_split(test_size=test_split)

            dataset = dataset.map(self.tokenize_function, batched=True)

            dataset = dataset.remove_columns(["text"])
            dataset.set_format("torch")

            # Loading test data
            if test_split:
                small_eval_dataset = dataset["test"]
                self.eval_data = DataLoader(small_eval_dataset, batch_size=batch_size)

            # Loading train data
            small_train_dataset = dataset["train"]
            self.train_data = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)


if __name__ == '__main__':
    tokenize = AutoTokenizer.from_pretrained("roberta-base")
    data = StringDataset(str(Path(data_location, "bank_account_classifier_training_set_processed.csv")), tokenize, ten_fold=True)
    print("eval batches:", len(data.eval_data[0]))
    print("train batches:", len(data.train_data[0]))
    for fold in data.train_data:
        for batch in fold:
            print(batch["labels"].shape, batch["input_ids"].shape, batch["attention_mask"].shape)
        break
