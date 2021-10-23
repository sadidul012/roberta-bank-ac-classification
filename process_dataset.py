from sklearn.preprocessing import LabelEncoder
import pandas as pd
from cutils import *


train_data = pd.read_csv(Path(data_location, "bank_account_classifier_training_set.csv"))

label_encoder = LabelEncoder()
label_encoder.fit(train_data["labels"])
save_object(label_encoder, "label_encoder")

train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:(1000*250)]

train_data["labels"] = label_encoder.transform(train_data["labels"])
train_data.to_csv(Path(data_location, "bank_account_classifier_training_set_processed.csv"), index=None)

test_data = pd.read_csv(Path(data_location, "bank_account_classifier_test_set.csv"))
test_data["labels"] = label_encoder.transform(test_data["labels"])
test_data.to_csv(Path(data_location, "bank_account_classifier_test_set_processed.csv"), index=None)

print("done")
