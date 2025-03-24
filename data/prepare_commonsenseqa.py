from datasets import load_dataset
import pandas as pd
import os

print("Preparing commonsenseqa...")
csqa = load_dataset("tau/commonsense_qa")
train_data = csqa['train'] # 9741 val for training
dev_data = csqa['validation'] # 1221 dev for dev

# To required format
answer_mapping = {label: index for index, label in enumerate(['A', 'B', 'C', 'D', 'E'])}
train_df = pd.DataFrame({
    'ID': [item['id'] for item in train_data],
    ## 'Question': [item['question'] + '\nChoices: [' + ', '.join([f"'{label}': '{text}'" for label, text in zip(item['choices']['label'], item['choices']['text'])]) + '].' for item in train_data], # concate choices['label'] with choices['text'], following a, b, c, d setting
    'Question': [item['question'] + '\nChoices: [' + ', '.join([f"{i}: '{choice}'" for i, choice in enumerate(item['choices']['text'])]) + '].' for item in train_data], # concate choice into question with index, following 1, 2, 3, 4 setting
    # 'Answers': [[str(item['answerKey'])] for item in train_data] # record answer as str
    'Answers': [[str(answer_mapping[item['answerKey']])] for item in train_data] # mapping char answer to int number
})

dev_df = pd.DataFrame({
    'ID': [item['id'] for item in dev_data],
    ## a, b, c, d setting
    ## 'Question': [item['question'] + '\nChoices: [' + ', '.join([f"'{label}': '{text}'" for label, text in zip(item['choices']['label'], item['choices']['text'])]) + '].' for item in dev_data], # concate choices['label'] with choices['text'], following a, b, c, d setting
    'Question': [item['question'] + '\nChoices: [' + ', '.join([f"{i}: '{choice}'" for i, choice in enumerate(item['choices']['text'])]) + '].' for item in dev_data], # concate choice into question with index, following 1, 2, 3, 4 setting
    # 'Answers': [[str(item['answerKey'])] for item in dev_data] # record answer as str
    'Answers': [[str(answer_mapping[item['answerKey']])] for item in dev_data] # mapping char answer to int number
})

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming train_df and dev_df are already defined as pandas DataFrames
train_file_path = os.path.join(script_dir, 'original', 'commonsenseqa_train.csv')
dev_file_path = os.path.join(script_dir, 'original', 'commonsenseqa_dev.csv')

# Save the datasets to CSV files
train_df.to_csv(train_file_path, index=True, header=True)
dev_df.to_csv(dev_file_path, index=True, header=True)

print(f"Training dataset size: {len(train_df)} rows")
print(f"Development dataset size: {len(dev_df)} rows")
print(f"Training dataset saved to: {train_file_path}")
print(f"Development dataset saved to: {dev_file_path}")