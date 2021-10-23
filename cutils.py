import pickle
from pathlib import Path

data_location = "/mnt/Cache/data/bank_account_classifier_data/bank_account_classifier_data"
num_classes = 125


def save_object(obj, obj_name):
    with open(Path(data_location, "{}.pkl".format(obj_name)), "wb") as file:
        pickle.dump(obj, file, protocol=3)


def load_object(obj_name):
    with open(Path(data_location, "{}.pkl".format(obj_name))) as file:
        return pickle.load(file)
