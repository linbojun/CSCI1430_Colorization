import os

# DIRECTORY INFORMATION
DATASET = "imagenet" # UPDATE
TEST_NAME ="FirstTest"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/'+DATASET+'/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/'+DATASET+'/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/'+DATASET+'/')
LOG_DIR = os.path.join(ROOT_DIR, 'LOGS/'+DATASET+'/')
OUT_DIR = os.path.abspath("../test_data/")
RESULT_DIR = os.path.abspath("../colorization_results/")

TRAIN_DIR = "train"  # UPDATE
TEST_DIR = "test" # UPDATE

# DATA
IMAGE_SIZE = 224
BATCH_SIZE = 10

# TRAIN
PRETRAINED = "modelPretrained.h5" # UPDATE
NUM_EPOCHS = 5