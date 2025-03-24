import pandas as pd
import os
import sys
import random
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_processing import write_divider


# Main function to join and reduce datasets
def join_data(name, dev_files, train_files):
    # Step 4: Load dev datasets
    #print("Reducing Dev Datasets:")
    dev_datasets = [pd.read_csv(file) for file in dev_files]
    
    # Combine dev datasets
    joint_dev = pd.concat(dev_datasets, ignore_index=True)

    # Load  train datasets
    #print("\nReducing Train Datasets:")
    train_datasets = [pd.read_csv(file) for file in train_files]

    # Combine train datasets
    joint_train = pd.concat(train_datasets, ignore_index=True)

    # Step 6: Save the reduced and concatenated datasets

    joint_dev.to_csv(f"data/training/{name}_generated_dev.csv", index=False)
    joint_train.to_csv(f"data/training/{name}_generated_train.csv", index=False)

    return joint_dev, joint_train
    
# Step 8: Command-line argument parsing
def main():
    parser = argparse.ArgumentParser(description="Join and reduce datasets for training and evaluation.")
    
    # Adding command-line arguments
    parser.add_argument('--experiment_name', type=str, required=True, help="Joint experiment name")
    parser.add_argument('--datasets', type=str, required=True, help="Comma-separated list of dataset identifiers (e.g., openbookqa,commonsenseqa)")

    args = parser.parse_args()

    # Parse input arguments
    name = args.experiment_name
    datasets = [dataset.strip() for dataset in args.datasets.split(',')]

    # Generate file paths for dev and train datasets based on names and datasets
    dev_files = [f"data/training/{name}_{dataset}_generated_dev.csv" for dataset in datasets]
    train_files = [f"data/training/{name}_{dataset}_generated_train.csv" for dataset in datasets]

    # Call the function to join data
    joint_dev, joint_train = join_data(name, dev_files, train_files)

    print(f"Joint dev size: {len(joint_dev)} rows")
    print(f"Joint train size: {len(joint_train)} rows")
    print("Datasets for DM training saved successfully")

    # Define the base directory for logs
    log_directory = "logs"

    # Ensure the logs directory exists (create if it doesn't exist)
    os.makedirs(log_directory, exist_ok=True)

    # Dynamic log filename based on the experiment or a specific naming pattern
    log_path = os.path.join(log_directory, f"{name}_log.txt")

    with open(log_path, 'a') as f:
        f.write(f"Joint dev size: {len(joint_dev)} rows\n")
        f.write(f"Joint train size: {len(joint_train)} rows\n")
        write_divider(f)

if __name__ == "__main__":
    main()