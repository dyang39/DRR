from datasets import load_dataset
import pandas as pd
import os

print("Preparing piqa...")
piqa = load_dataset("ybisk/piqa")
train_data = piqa['train'] # 16113 val for training
dev_data = piqa['validation'] # 1838 val for dev
# test_data = piqa['test'] # 3084 test for test

# To required format
train_df = pd.DataFrame({
    'ID': range(len(train_data)),
    'Question': [item['goal'] + "\nChoices: ['0': '" + item['sol1'] + "', '1': '" + item['sol2'] + "']." for item in train_data],
    'Answers': [[str(item['label'])] for item in train_data]
})

dev_df = pd.DataFrame({
    'ID': range(len(dev_data)),
    'Question': [item['goal'] + "\nChoices: ['0': '" + item['sol1'] + "', '1': '" + item['sol2'] + "']." for item in dev_data],
    'Answers': [[str(item['label'])] for item in dev_data]
})

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming train_df and dev_df are already defined as pandas DataFrames
train_file_path = os.path.join(script_dir, 'original', 'piqa_train.csv')
dev_file_path = os.path.join(script_dir, 'original', 'piqa_dev.csv')

# Save the datasets to CSV files
train_df.to_csv(train_file_path, index=True, header=True)
dev_df.to_csv(dev_file_path, index=True, header=True)

print(f"Training dataset size: {len(train_df)} rows")
print(f"Development dataset size: {len(dev_df)} rows")
print(f"Training dataset saved to: {train_file_path}")
print(f"Development dataset saved to: {dev_file_path}")