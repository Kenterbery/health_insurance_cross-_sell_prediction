import os

DATA_FOLDER = "data"
ESTIMATOR_FOLDER = "models"
TRAIN_CSV = os.path.join(DATA_FOLDER, "train.csv")
VAL_CSV = os.path.join(DATA_FOLDER, "val.csv")
SAVED_ESTIMATOR = os.path.join(ESTIMATOR_FOLDER, "log_reg.pickle")