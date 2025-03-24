import sys
import argparse
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from utils.prompt_template import *
import re
import json
import pandas as pd
from utils.evaluation import *
from utils.text_processing import safe_literal_eval, extract_final_answer_and_rationale, extract_assistant_output, extract_final_answer, write_divider
import os
import ast
import time
from sklearn.metrics import accuracy_score
from accelerate import Accelerator
import openai

def call_reasoner_batch(task_content,temperature=0,top_p=0.9):
    # Prepare messages for all batches
    message = [
        {"role": "user", "content": task_content} 
    ]
    # Make API call to OpenAI for batch processing
    response = openai.chat.completions.create(
        model=MODEL,  # Specify the GPT-4 model
        messages=message,
        temperature=temperature,
        max_tokens=256,  
        stop=None
    )

    return response.choices[0].message.content


def parse_answer(response):
    """
    Extract the final answer number from a response string. The answer is expected 
    in the form '(number)' at the end of the response. If not found, return the last 
    number in the string as a fallback.

    Args:
        response (str): The response string containing the final answer in the form '(number)'.

    Returns:
        int or None: The extracted number as an integer, or None if no number is found.
    """
    # Look for a number inside parentheses at the end of the response
    match = re.search(r"\((\d+)\)\s*$", response)
    if match:
        return int(match.group(1))
    
    # Fallback: Find all numbers and return the last one
    all_numbers = re.findall(r"\d+", response)
    if all_numbers:
        return int(all_numbers[-1])
    
    # If no number is found, return None
    return -1

def main(task_content):
    task_content = task_content + ("Explain your reasoning. You must choose only one option from above. Your "
        "final answer should be a single number (e.g., 0, 1, 2, etc), in the form (answer), at the end of your response.")

    response1 = call_reasoner_batch(task_content)
    answer1 = parse_answer(response1)
    print(f"Answer 1: {answer1}")
    task_content = task_content + "\n" + response1 + "\n" + ("Review your previous answer and find problems with your answer.")
    
    response2 = call_reasoner_batch(task_content)

    task_content = task_content + "\n" + response2 + "\n" + ("Based on the problems you found, improve your answer. You must choose "
        "only one option from above. Please reiterate your answer, with your "
        "final answer a single number (e.g., 0, 1, 2, etc), in the form (answer).")
    
    response3 = call_reasoner_batch(task_content)
    answer2 = parse_answer(response3)
    print(f"Answer 2: {answer2}")
    task_content = task_content + "\n" + response3 + "\n" +  ("Review your previous answer and find problems with your answer.")
    
    response4 = call_reasoner_batch(task_content)

    task_content = task_content + "\n" + response4 + "\n" + ("Based on the problems you found, improve your answer. You must choose "
        "only one option from above. Please reiterate your answer, with your "
        "final answer a single number (e.g., 0, 1, 2, etc), in the form (answer).")
    
    response5 = call_reasoner_batch(task_content)
    answer3 = parse_answer(response5)
    print(f"Answer 3: {answer3}")
    return answer1, answer2, answer3, task_content


def evaluate_and_save_results(json_file, output_file, log_file, question_type, abstain):
    def save_results_to_csv(df, filename):
        """Save results to CSV, creating file if it doesn't exist."""
        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)

    # Load validation data
    validation_data = pd.read_csv(json_file)

    results = []

    # Track the accuracies and formula scores
    correct_answer_1 = []
    correct_answer_2 = []
    correct_answer_3 = []
    formula_scores_1 = []
    formula_scores_2 = []
    formula_scores_3 = []

    for i, entry in validation_data.iterrows():
        task_content = f"Q: {entry['Question']}\n"

        print(i)
        # Get the predicted answers from the model
        answer1, answer2, answer3, task_content = main(task_content)

        # Normalize answers
        answer1 = normalize_answer(str(answer1).lower())
        answer2 = normalize_answer(str(answer2).lower())
        answer3 = normalize_answer(str(answer3).lower())

        # Check correctness for all three answers
        correct1 = exact_match_multiple(answer1, safe_literal_eval(entry['Answers']))
        correct2 = exact_match_multiple(answer2, safe_literal_eval(entry['Answers']))
        correct3 = exact_match_multiple(answer3, safe_literal_eval(entry['Answers']))

        # Append results for the current entry
        results.append({
            'ID': entry['ID'],
            'Question': entry['Question'],
            'Gold Answer': entry['Answers'],
            'Answer 1': answer1,
            'Answer 2': answer2,
            'Answer 3': answer3,
            'Correct Answer 1': correct1,
            'Correct Answer 2': correct2,
            'Correct Answer 3': correct3,
            'Task Content File': task_content  # Reference to task content file
        })

        # Track accuracy and formula scores for each answer
        correct_answer_1.append(correct1)
        correct_answer_2.append(correct2)
        correct_answer_3.append(correct3)

        # Save intermediate results after each entry
        intermediate_df = pd.DataFrame(results)
        save_results_to_csv(intermediate_df, output_file)

    results_df = pd.DataFrame(results)
    save_results_to_csv(results_df, output_file)

    # Calculate accuracy for each answer
    accuracy_1 = accuracy_score(correct_answer_1, [1] * len(correct_answer_1))
    accuracy_2 = accuracy_score(correct_answer_2, [1] * len(correct_answer_2))
    accuracy_3 = accuracy_score(correct_answer_3, [1] * len(correct_answer_3))

    formula_df1 = results_df.rename(columns={'Correct Answer 1': 'Correct Answer','Answer 1': 'Reasoner Answer'})
    formula_df2 = results_df.rename(columns={'Correct Answer 2': 'Correct Answer','Answer 2': 'Reasoner Answer'})
    formula_df3 = results_df.rename(columns={'Correct Answer 3': 'Correct Answer','Answer 3': 'Reasoner Answer'})
    # Calculate formula score for the entire DataFrame

    formula_1 = get_formula_score(formula_df1, zero_shot=True)
    formula_2 = get_formula_score(formula_df2, zero_shot=True)
    formula_3 = get_formula_score(formula_df3, zero_shot=True)

    # Log accuracy and formula scores for each answer
    with open(log_file, 'a') as log_f:
        log_f.write(f"Zero Shot Accuracy (Answer 1): {accuracy_1:.4f}\n")
        log_f.write(f"Zero Shot Accuracy (Answer 2): {accuracy_2:.4f}\n")
        log_f.write(f"Zero Shot Accuracy (Answer 3): {accuracy_3:.4f}\n")
        # Print and save the formula scores for each answer
        log_f.write(f"Formula Score (Answer 1): {formula_1:.4f}\n")
        log_f.write(f"Formula Score (Answer 2): {formula_2:.4f}\n")
        log_f.write(f"Formula Score (Answer 3): {formula_3:.4f}\n")
        
        write_divider(f)

    print(f"Accuracy Score (Answer 1): {accuracy_1:.4f}")
    print(f"Accuracy Score (Answer 2): {accuracy_2:.4f}")
    print(f"Accuracy Score (Answer 3): {accuracy_3:.4f}")
    print(f"Formula Score (Answer 1): {formula_1:.4f}\n")
    print(f"Formula Score (Answer 2): {formula_2:.4f}\n")
    print(f"Formula Score (Answer 3): {formula_3:.4f}\n")

    print(f"Evaluation complete. Results saved to {output_file}.")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Evaluate the model on validation data in a zero shot manner, with no DAI model.')

    # Define command-line arguments
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment, used for naming and paths.')
    parser.add_argument('--input_path', type=str, required=False, help='Path to the input dataset CSV file. Default is generated using the experiment name.')
    parser.add_argument('--output_path', type=str, required=False, help='Path to save the output CSV file. Default is generated using the experiment name.')
    parser.add_argument('--log_path', type=str, required=False, help='Path to save the log file. Default is generated using the experiment name.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to process at a time.')
    parser.add_argument('--question_type', type=str, choices=['MCQ', 'OEQ','MATH'], default='OEQ', help='Type of question: MCQ for multiple choice, OEQ for open-ended, MATH for open-ended math.')
    parser.add_argument('--abstain', action="store_true", required=False, help='Allows zero shot prediction which gives LLM option to abstain.')
    parser.add_argument('--gpt_model', type=str, required=False, help='GPT model to use, e.g., "gpt-4" or "gpt-3.5".')

    # Parse the arguments
    args = parser.parse_args()

    if args.input_path is None:
        args.input_path = f'data/original/{args.experiment_name}_dev.csv'
    if args.output_path is None:
        args.output_path = f'predictions/{args.experiment_name}_predictions.csv'
    if args.log_path is None:
        args.log_path = f'logs/{args.experiment_name}_selfcritic_log.txt'



    # Ensure OpenAI API key is provided
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key must be provided via environment variable.")
    openai.api_key = openai_api_key
    MODEL = args.gpt_model if args.gpt_model else "gpt-4o-mini-2024-07-18"

    # Call the evaluate function with the provided paths
    evaluate_and_save_results(args.input_path, args.output_path, args.log_path, args.question_type, args.abstain)