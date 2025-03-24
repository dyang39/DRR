import pandas as pd
import os
import argparse
from utils.evaluation import get_accuracy, get_formula_score  

def evaluate_model(file_path, zero_shot):
    # Construct the file path for the predictions file
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Load the dataframe from the predictions file
    df = pd.read_csv(file_path)
    df_zs = df.groupby('ID').head(1)

    # Run get_accuracy on the dataframe
    accuracy = get_accuracy(df)  
    formula_score = get_formula_score(df)
    zs_accuracy = get_accuracy(df_zs)  
    zs_formula_score = get_formula_score(df_zs,zero_shot=True)

    print(f"ZS Accuracy: {accuracy:.4f}")
    print(f"ZS Formula Score: {formula_score:.4f}")
    print(f"DRR Accuracy: {accuracy:.4f}")
    print(f"DRR Formula Score: {formula_score:.4f}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate the model's accuracy and formula score based on predictions.")
    
    # Add argument for the model and dataset name
    parser.add_argument('file_path', type=str, 
                        help="File path to the predictions file")

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the evaluate_model function with the input argument
    evaluate_model(args.file_path)

if __name__ == "__main__":
    main()
