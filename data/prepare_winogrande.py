from datasets import load_dataset
import pandas as pd
import os
import random

print("Preparing winogrande...")
wino = load_dataset("allenai/winogrande", "winogrande_l")
train_data = wino['train'] # 10234 val for training
dev_data = wino['validation'] # 1267 val for dev

# shuffle train data
train_data_list = list(train_data)
random.shuffle(train_data_list)

# Prepare train and dev DataFrames
train_df = pd.DataFrame({
    'ID': [str(i) for i in range(len(train_data_list))],
    'Question': [item['sentence'].replace("_", "[MASK]") + '\nChoices: [0: \'' + item['option1'] + '\', 1: \'' + item['option2'] + '\'].' for item in train_data_list],
    'Answers': [[str(int(item['answer']) - 1)] for item in train_data_list]  # Convert answer to 0-based index in a list format
})

dev_df = pd.DataFrame({
    'ID': [str(i) for i in range(len(dev_data))],
    'Question': [item['sentence'].replace("_", "[MASK]") + '\nChoices: [0: \'' + item['option1'] + '\', 1: \'' + item['option2'] + '\'].' for item in dev_data],
    'Answers': [[str(int(item['answer']) - 1)] for item in dev_data]  # Convert answer to 0-based index in a list format
})

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming train_df and dev_df are already defined as pandas DataFrames
train_file_path = os.path.join(script_dir, 'original', 'winogrande_train.csv')
dev_file_path = os.path.join(script_dir, 'original', 'winogrande_dev.csv')

# Save the datasets to CSV files
train_df.to_csv(train_file_path, index=True, header=True)
dev_df.to_csv(dev_file_path, index=True, header=True)

print(f"Training dataset size: {len(train_df)} rows")
print(f"Development dataset size: {len(dev_df)} rows")
print(f"Training dataset saved to: {train_file_path}")
print(f"Development dataset saved to: {dev_file_path}")