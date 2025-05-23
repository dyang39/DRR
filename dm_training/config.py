# Model configuration
MODEL_PATH = "google/flan-t5-large"
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 64

# Data configuration
EXPERIMENT_NAME = "example"

# Training configuration
#OUTPUT_DIR = "finetuning_output/"
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
NUM_TRAIN_EPOCHS = 3 #3?
WARMUP_STEPS = 500
LOGGING_STEPS = 500
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 3
FP16 = False