import argparse
import sys
import os
import re
import json
import time
import math
import ast
import torch
import openai
import pandas as pd
from sklearn.metrics import accuracy_score
from accelerate import Accelerator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_template import *
from utils.evaluation import *
from utils.text_processing import (
    safe_literal_eval, extract_final_answer_and_rationale, extract_assistant_output, write_divider
)

def call_reasoner_batch(system_prefixes, task_contents, gpt, temperature = 0.1, top_p = 0.9):
    """ Calls the reasoner (GPT or Llama) in batch mode."""
    # Prepare messages for all batches
    if isinstance(system_prefixes, str):
        messages = [
            [
            {"role": "system", "content": system_prefixes},
            {"role": "user", "content": task_content} 
            ] for task_content in task_contents 
        ]
    else:    
        messages = [
            [
            {"role": "system", "content": system_prefix},
            {"role": "user", "content": task_content}
            ] for task_content, system_prefix in zip(task_contents, system_prefixes)
        ]

    #Call GPT
    if gpt:
        responses = []
        for message in messages:
        
            # Make API call to OpenAI
            response = openai.chat.completions.create(
                model=MODEL, 
                messages=message,
                temperature=temperature,
                max_tokens=256,  
                stop=None
            )

            responses.append(response.choices[0].message.content)
        return responses
    #Call Llama
    else:
        #Preparation
        messages = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(messages, padding="longest", return_tensors="pt")
        inputs = {key: val.to(accelerator.device) for key, val in inputs.items()}  # Move to correct device
        
        # Batch generation
        outputs_batch = model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
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

def check_accept_or_reject(batch_responses, batch_gold_answers):
    """Compares model responses with gold answers."""
    return ["1" if response in answers else "0" for response, answers in zip(batch_responses, batch_gold_answers)]

def save_results_to_csv(df, filename):
    """Saves DataFrame to CSV, appending if the file exists."""
    df.to_csv(filename, mode='a' if os.path.isfile(filename) else 'w', header=not os.path.isfile(filename), index=False)

def main_batch(task_contents, question_type, ids, batch_gold_answers, checkpoint_file, max_turns, force_rationale, prompt_setting, gpt):
    #Gradual prompt setting
    if prompt_setting == 'gradual':
        turn = 1
        results = []
        previous_answers = [""] * len(task_contents)
        
        max_iter = max_turns + 1 if force_rationale else max_turns

        while turn <= max_iter:
            if turn == max_turns+1:
                # Set prompt based on current turn and question type
                assert force_rationale == True
                if question_type == "MCQ":
                    prompt = mcq_force_rationale_prompt 
                else:
                    prompt = [create_force_rationale_prompt(batch_gold_answer[0], question_type) for batch_gold_answer in batch_gold_answers]
            else:
                if question_type == 'MCQ':
                    prompt = mcq_force_answer_prompt_gradual
                elif question_type == 'MATH':
                    prompt = math_force_answer_prompt_gradual
                else:
                    prompt = force_answer_prompt_gradual
            batch_responses = call_reasoner_batch(prompt, task_contents, gpt, 0.6)
            answers, rationales = zip(*[extract_final_answer_and_rationale(response, question_type) for response in batch_responses])

            verdicts = []
            for i in range(len(answers)):
                verdicts.append(exact_match_multiple(answers[i],batch_gold_answers[i]))

            remaining_task_indices = []
            for i, (answer, rationale, verdict) in enumerate(zip(answers, rationales, verdicts)):
                results.append({
                    'ID': ids[i],
                    'Turn': turn,
                    'Reasoner Task Content': task_contents[i],
                    'Reasoner Answer': normalize_answer(answer),
                    'Reasoner Rationale': rationale, 
                    'Gold Answer': batch_gold_answers[i],
                    'Verdict': verdict
                    })
                if verdict == 0:
                    remaining_task_indices.append(i)
            
            if len(remaining_task_indices) == 0:
                break

            # Filter the tasks that still need to be processed
            task_contents = [task_contents[i] for i in remaining_task_indices]
            ids = [ids[i] for i in remaining_task_indices]
            previous_answers = [previous_answers[i] for i in remaining_task_indices]
            answers = [answers[i] for i in remaining_task_indices]
            rationales = [rationales[i] for i in remaining_task_indices]
            verdicts = [verdicts[i] for i in remaining_task_indices]
            batch_gold_answers = [batch_gold_answers[i] for i in remaining_task_indices]
            batch_responses = [batch_responses[i] for i in remaining_task_indices]
            
            for i, (answer, rationale) in enumerate(zip(answers, rationales)):
                answer = answer if answer is not None else ""
                rationale = rationale if rationale is not None else "" 

                if turn < max_turns:
                    if exact_match_multiple(answer, [previous_answers[i]]):
                        task_contents[i] = task_contents[i] + f"LLM Answer {turn}: " + answer + f"\nLLM Rationale {turn}: " + rationale + f"\n{exact_wrong_prompt}\n"
                    else:
                        task_contents[i] = task_contents[i]  + f"LLM Answer {turn}: " + answer + f"\nLLM Rationale {turn}: " + rationale + f"\n{feedback_prompt}\n"
                else:
                    if question_type == "MCQ":
                        task_contents[i] = task_contents[i] + "\nGround-truth Answer: " + batch_gold_answers[i][0] + "\n"
                    else: 
                        task_contents[i] = task_contents[i] + "\nUse this exact correct answer: " + batch_gold_answers[i][0] + "\n"

            previous_answers = answers
            turn += 1

        save_df = pd.DataFrame(results)
        save_results_to_csv(save_df, checkpoint_file)

        return results

    # Exploration prompt setting
    else:    
        cur = 0
        results = []

        max_iter = max_turns + 1 if force_rationale else max_turns

        while cur <= max_iter:
            # Set prompt based on current turn and question type
            if cur == 0:
                if question_type == 'MCQ':
                    prompt = mcq_force_answer_prompt 
                elif question_type == 'MATH':
                    prompt = math_force_answer_prompt
                else:
                    prompt = force_answer_prompt 
                batch_responses = call_reasoner_batch(prompt, task_contents, gpt)
            elif cur == max_turns+1: #Force-Rationale
                if question_type == "MCQ":
                    prompt = mcq_force_rationale_prompt 
                    batch_responses = call_reasoner_batch(prompt, task_contents, gpt)
                else: 
                    force_rationale_prompts = [create_force_rationale_prompt(batch_gold_answer[0], question_type) for batch_gold_answer in batch_gold_answers]
                    batch_responses = call_reasoner_batch(force_rationale_prompts, task_contents, gpt)
            else:
                if question_type == 'MCQ':
                    prompt = mcq_exploration_force_answer_prompt 
                elif question_type == 'MATH':
                    prompt = math_exploration_force_answer_prompt
                else:
                    prompt = exploration_force_answer_prompt
                batch_responses = call_reasoner_batch(prompt, task_contents, gpt, 0.6, 0.7)
            answers, rationales = zip(*[extract_final_answer_and_rationale(response, question_type) for response in batch_responses])

            verdicts = []
            for i in range(len(answers)):
                verdicts.append(exact_match_multiple(answers[i],batch_gold_answers[i]))

            remaining_task_indices = []
            for i, (answer, rationale, verdict) in enumerate(zip(answers, rationales, verdicts)):
                results.append({
                    'ID': ids[i],
                    'Turn': cur,
                    'Reasoner Task Content': task_contents[i],
                    'Reasoner Answer': normalize_answer(answer),
                    'Reasoner Rationale': rationale, 
                    'Gold Answer': batch_gold_answers[i],
                    'Verdict': verdict
                    })
                if verdict == 0:
                    remaining_task_indices.append(i)
            
            if (len(remaining_task_indices) == 0):
                break

            # Filter the tasks that still need to be processed
            task_contents = [task_contents[i] for i in remaining_task_indices]
            ids = [ids[i] for i in remaining_task_indices]
            answers = [answers[i] for i in remaining_task_indices]
            rationales= [rationales[i] for i in remaining_task_indices]
            verdicts = [verdicts[i] for i in remaining_task_indices]
            batch_gold_answers = [batch_gold_answers[i] for i in remaining_task_indices]
            batch_responses = [batch_responses[i] for i in remaining_task_indices]

            for i in range(len(task_contents)):
                if cur < max_turns:
                    task_contents[i] = task_contents[i] + "Previous LLM Response: " + batch_responses[i] + "\nWrong answer! Try again.\n"
                else:
                    if question_type == "MCQ":
                        task_contents[i] = task_contents[i] + "\nGround-truth Answer: " + batch_gold_answers[i][0] + "\n"
                    else:
                        task_contents[i] = task_contents[i] + "\nUse this exact correct answer: " + batch_gold_answers[i][0] + "\n"

            cur += 1

        save_df = pd.DataFrame(results)
        save_results_to_csv(save_df, checkpoint_file)
        return results

def run_system(input_path, output_path, question_type, log_path, start=0, end=10000, 
               batch_size=8, max_turns=2, force_rationale=True, prompt_setting='gradual', gpt=False):
    start_time = time.time()
    checkpoint_file = output_path

    # Load the original data
    original_data = pd.read_csv(input_path)
    end = min(end, len(original_data))

    # Log file setup
    with open(log_path, "a") as f:
        f.write(f"Original Dataset Size: {end-start}\n")
        f.write(f"Question Type: {question_type}, Force Rationale: {force_rationale}, Prompt Setting: {prompt_setting}, GPT: {gpt}\n")
        write_divider(f)

    # Process the data in batches
    for i in range(start, end, batch_size):
        batch = original_data.iloc[i:i + batch_size]
        batch_ids = batch['ID'].tolist()
        batch_questions = batch['Question'].tolist()
        batch_gold_answers = batch['Answers'].apply(safe_literal_eval).tolist()
        task_contents = [f"Q: {q} \n" for q in batch_questions]

        batch_predicted_answers = main_batch(
            ids=batch_ids,
            task_contents=task_contents,
            question_type=question_type,
            batch_gold_answers=batch_gold_answers,
            checkpoint_file=checkpoint_file,
            max_turns=max_turns,
            force_rationale=force_rationale,
            prompt_setting=prompt_setting,
            gpt=gpt
        )

        print(f"Processed batch {i // batch_size + 1}/{(len(original_data) + batch_size - 1) // batch_size}")

    # Log completion and statistics
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Generation complete. Results saved to CSV. Elapsed Time: {elapsed_time:.2f} seconds")

    # Save the current stdout
    original_stdout = sys.stdout

    # Redirect stdout to null (suppress print output)
    sys.stdout = open(os.devnull, 'w')
    df = pd.read_csv(output_path)
    with open(log_path, "a") as f:
        f.write("Generated Data Statistics\n")
        turn_counts, zero_shot_accuracy, f1 = per_turn_generation(df, prompt_setting)
        f.write(f"Verdict Distribution per Turn (Generated Data): {turn_counts}\n")
        f.write(f"Zero Shot Accuracy for Turn = 0 (Generated Data): {zero_shot_accuracy:.2f}\n")
        f.write(f"Zero Shot F1 for Turn = 0 (Generated Data): {f1:.2f}\n")
        f.write(f"System Accuracy (Generated Data): {get_accuracy_generation(df):.4f}\n")
        write_divider(f)
    # Restore the original stdout
    sys.stdout = original_stdout


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Generate data for DAI model training through simulating the system during inference.')

    # Define command-line arguments
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name used to save files. This name will also be used to generate default paths for input, output, and log files.')
    parser.add_argument('--input_path', type=str, required=False, help='Path to the input dataset CSV file. Default is generated using the experiment name.')
    parser.add_argument('--output_path', type=str, required=False, help='Path to the output dataset CSV file. Default is generated using the experiment name.')
    parser.add_argument('--log_path', type=str, required=False, help='Path to the log file. Default is generated using the experiment name.')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for the original dataset.')
    parser.add_argument('--end_index', type=int, default=10000, help='End index for the original dataset.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size to process at a time.')
    parser.add_argument('--max_turns', type=int, help='Max turns of responses, starting at 0.')
    parser.add_argument('--question_type', type=str, choices=['MCQ', 'OEQ', 'MATH'], default='MCQ', help='Type of question: MCQ for multiple choice, OEQ for open-ended, MATH for open-ended math.')
    parser.add_argument('--force_rationale', action="store_true", default=False, help='If passed, enables forcing rationale generation at max_turns + 1.')
    parser.add_argument('--prompt_setting', type=str, choices=['gradual', 'exploration'], default='gradual', help='Type of prompt setting: "gradual" or "exploration".')
    parser.add_argument('--gpt', action="store_true", default=False, help='Use llama (default) or GPT if passed in.')
    parser.add_argument('--gpt_model', type=str, required=False, help='GPT model to use, e.g., "gpt-4" or "gpt-3.5".')
    parser.add_argument('--llama_model', type=str, required=False, help='Path to the Llama model if GPT is not used.')

    # Parse the arguments
    args = parser.parse_args()

    # Set default for input_path based on experiment_name if not provided
    if args.input_path is None:
        args.input_path = f'data/original/{args.experiment_name}_train.csv'
    if args.output_path is None:
        args.output_path = f'data/generated/{args.experiment_name}_generated.csv'
    if args.log_path is None:
        args.log_path = f'logs/{args.experiment_name}_log.txt'

    # Dynamically set max_turns based on prompt_setting if not explicitly provided
    if args.max_turns is None:
        args.max_turns = 3 if args.prompt_setting == 'exploration' else 4

    # Initialize the accelerator 
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    torch.cuda.empty_cache()
    accelerator = Accelerator()

    ## LLM ##
    from transformers import AutoTokenizer, AutoModelForCausalLM
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

    # Call the run_system function with parsed arguments
    run_system(args.input_path, args.output_path, args.question_type, args.log_path, args.start_index, args.end_index, args.batch_size, args.max_turns, args.force_rationale, args.prompt_setting, args.gpt)

