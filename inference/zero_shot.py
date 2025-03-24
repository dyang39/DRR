import sys
import argparse
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
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

def call_reasoner_batch(system_prefix, task_contents,gpt,temperature=0.1,top_p=0.9):
    # Prepare messages for all batches
    messages = [
        [
            {"role": "system", "content": system_prefix},
            {"role": "user", "content": task_content} 
        ] for task_content in task_contents # Prepare messages for all batches
    ]
    if gpt:
        responses = []
        for message in messages:
        
            # Make API call to OpenAI for batch processing
            response = openai.chat.completions.create(
                model=MODEL,  # Specify the GPT-4 model
                messages=message,
                temperature=temperature,
                max_tokens=256,  
                stop=None
            )

            responses.append(response.choices[0].message.content)
        return responses
    else:
        messages = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(messages, padding="longest", return_tensors="pt")
        #inputs = {key: val.to(model.device) for key, val in inputs.items()} # Cache the V
        inputs = {key: val.to(accelerator.device) for key, val in inputs.items()}  # Move to correct device
        
        # Batch generation
        outputs_batch = model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],  # Apply the attention mask as well
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        # Batch decoding
        responses = tokenizer.batch_decode(
            outputs_batch, 
            skip_special_tokens=True
        )

        # Extract the assistant's output from each response
        extracted_outputs = [extract_assistant_output(response) for response in responses]
        
        return extracted_outputs

def main(task_contents, question_type, abstain, gpt):
    if question_type == 'MCQ':
        if abstain:
            prompt = abstain_mcq_force_answer_prompt
        else:
            prompt = mcq_force_answer_prompt
    elif question_type == 'MATH':
        if abstain:
            prompt = abstain_math_force_answer_prompt
        else:
            prompt = math_force_answer_prompt
    else:
        if abstain:
            prompt = abstain_force_answer_prompt
        else:
            prompt = force_answer_prompt 

    responses = call_reasoner_batch(prompt, task_contents, gpt)

    return [extract_final_answer(response,question_type) for response in responses]

def evaluate_and_save_results_batch(json_file, output_file, log_file, batch_size, question_type, abstain, gpt):
    """
    Evaluates the model on the validation data in batches, saves results to a CSV file, 
    and logs the accuracy.

    Parameters:
    - json_file: Path to the JSON file containing validation data.
    - output_file: Path to the CSV file where results will be saved.
    - model_func: Function to get the model's predicted answer given the task content.
    - batch_size: Number of samples to process in each batch.
    - question_type: Type of question - MCQ for multiple choice, OEQ for open-ended.
    """
    def save_results_to_csv(df, filename):
        """Save results to CSV, creating file if it doesn't exist."""
        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)

    start_time = time.time()

    # Load validation data
    validation_data = pd.read_csv(json_file)


    # Process data in batches
    all_results = []  # Track results of all batches; temporary solution - memory inefficient

    for batch_start in range(0, len(validation_data), batch_size): #len(validation_data)
        batch_end = batch_start + batch_size
        batch_data = validation_data.iloc[batch_start:batch_end]
        
        batch_results = []

        # Prepare batch task contents
        task_contents = [f"Question: {row['Question']}\n" for _, row in batch_data.iterrows()]

        #try:
            # Get the predicted answers for the batch
        predicted_answers = main(task_contents, question_type, abstain, gpt)
        #except Exception as e:
        #    predicted_answers = [str(e)] * len(batch_data)

        # Process each entry in the batch
        for i, (_, entry) in enumerate(batch_data.iterrows()):
            predicted_answer = predicted_answers[i].lower()

            # Add results to list
            batch_results.append({
                'ID': entry['ID'],
                'Question': entry['Question'],
                'Gold Answer': entry['Answers'],
                'Reasoner Answer': normalize_answer(predicted_answer),
                'Correct Answer': exact_match_multiple(predicted_answer, safe_literal_eval(entry['Answers']))
            })

        all_results.extend(batch_results)  # update to track results for all batch

        print(f"Processed batch {batch_start // batch_size + 1}/{(len(validation_data) + batch_size - 1) // batch_size}")

        # Convert results to DataFrame
        results_df = pd.DataFrame(batch_results)

        # Save all results to CSV at once
        save_results_to_csv(results_df, output_file)

    end_time = time.time()
    elapsed_time = end_time - start_time

    all_df = pd.read_csv(output_file)
    #calculate formula score
    formula_score = get_formula_score(all_df, zero_shot = True)
    with open(log_file,'a') as f:
        f.write(f"Zero Shot Formula Score: {formula_score:.4f}\n")
        write_divider(f)
    
    # Calculate accuracy
    system_accuracy = accuracy_score(all_df['Correct Answer'], [1] * len(all_df))
    f1 = get_f1_score(all_df)

    # Print results
    with open(log_file,'a') as f:
        f.write(f"Zero Shot Accuracy: {system_accuracy:.4f}\n")
        f.write(f"Zero Shot F1: {f1:.4f}\n")
        write_divider(f)

    print(f"Accuracy Score: {system_accuracy:.4f}")
    print(f"F1: {f1:.4f}\n")
    print(f"Evaluation complete. Results saved to {output_file}.")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Evaluate the model on validation data in a zero shot manner, with no DAI model.')

    # Define command-line arguments
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment, used for naming and paths.')
    parser.add_argument('--input_path', type=str, required=False, help='Path to the input dataset CSV file. Default is generated using the experiment name.')
    parser.add_argument('--output_path', type=str, required=False, help='Path to save the output CSV file. Default is generated using the experiment name.')
    parser.add_argument('--log_path', type=str, required=False, help='Path to save the log file. Default is generated using the experiment name.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to process at a time.')
    parser.add_argument('--question_type', type=str, choices=['MCQ', 'OEQ','MATH'], default='MCQ', help='Type of question: MCQ for multiple choice, OEQ for open-ended, MATH for open-ended math.')
    parser.add_argument('--gpt', action="store_true", default=False, help='Use llama (default) or GPT if pass in True.')
    parser.add_argument('--gpt_model', type=str, required=False, help='GPT model to use, e.g., "gpt-4" or "gpt-3.5".')
    parser.add_argument('--llama_model', type=str, required=False, help='Path to the Llama model if GPT is not used.')
    parser.add_argument('--abstain', action="store_true", required=False, help='Allows zero shot prediction which gives LLM option to abstain.')

    # Parse the arguments
    args = parser.parse_args()

    if args.input_path is None:
        args.input_path = f'data/original/{args.experiment_name}_dev.csv'
    if args.output_path is None:
        args.output_path = f'predictions/{args.experiment_name}_predictions.csv'
    if args.log_path is None:
        args.log_path = f'logs/{args.experiment_name}_log.txt'

    # Load your model function, assuming it's defined in your_model_module
    accelerator = Accelerator()

    ## LLM ##
    if args.gpt:
        # Ensure OpenAI API key is provided
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided via environment variable.")
        openai.api_key = openai_api_key
        MODEL = args.gpt_model if args.gpt_model else "gpt-4o-mini-2024-07-18"
    else:
        if not args.llama_model:
            raise ValueError("Llama model path must be provided when not using GPT.")
        model_id = args.llama_model
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        model, tokenizer = accelerator.prepare(model, tokenizer)

    # Call the evaluate function with the provided paths
    evaluate_and_save_results_batch(args.input_path, args.output_path, args.log_path, args.batch_size, args.question_type, args.abstain, args.gpt)