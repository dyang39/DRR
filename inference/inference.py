#No retrieval version for triviaQA
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

def call_reasoner_batch(system_prefix, task_contents, gpt, temperature=0.1,top_p=0.9):
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


def call_gate_batch(answers, rationales, task_contents, gate_model, gate_tokenizer, weighted_training, prompt_setting = 'exploration'):
    instruction = "Instruction: Predict if the following answer to the question and context should be accepted, 1, or rejected, 0, based on the rationale."
    if prompt_setting == 'gradual':
        gate_inputs = [
        f"{instruction}\n{task_content} \nLLM Answer: {answer}\nLLM Rationale: {rationale}"
        for answer, rationale, task_content in zip(answers, rationales, task_contents)
        ]
    else:
        gate_inputs = [
        f"{instruction}\n{task_content} \nAnswer: {answer}\nRationale: {rationale}"
        for answer, rationale, task_content in zip(answers, rationales, task_contents)
        ]

    # to accelerator device
    inputs = gate_tokenizer(gate_inputs, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
    
    if weighted_training:
        generated_ids = gate_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=1  # only need one new token
        )
        # responses = gate_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        responses = gate_tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=True)
        # Get logits
        last_token_logits = generated_ids.scores[0]  # shape: [batch_size, vocab_size]
        # Apply softmax for probs
        probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
        
        # Get token id for "0" and "1" 
        token_0_id = gate_tokenizer.encode("0", add_special_tokens=False)[0]
        token_1_id = gate_tokenizer.encode("1", add_special_tokens=False)[0]
        
        # Get probs
        probs_0 = probs[:, token_0_id].cpu().numpy()
        probs_1 = probs[:, token_1_id].cpu().numpy()
        
        predictions = [response == '1' for response in responses]
        confidences = {
            'predictions': predictions,
            'confidence_0': probs_0,
            'confidence_1': probs_1
        }
        return confidences 
    else:
        generated_ids = gate_model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )

        responses = gate_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [response == '1' for response in responses]

def save_results_to_csv(df, filename):
    """Saves DataFrame to CSV, appending if the file exists."""
    df.to_csv(filename, mode='a' if os.path.isfile(filename) else 'w', header=not os.path.isfile(filename), index=False)

def main_batch(task_contents, question_type, ids, batch_gold_answers, checkpoint_file, max_turns, weighted_training, prompt_setting, gpt):
    #Gradual prompt setting
    if prompt_setting == 'gradual':
        turn = 1
        results = []
        previous_answers = [""] * len(task_contents)

        while turn <= max_turns:
            if question_type == 'MCQ':
                prompt = mcq_force_answer_prompt_gradual
            elif question_type == 'MATH':
                prompt = math_force_answer_prompt_gradual
            else:
                prompt = force_answer_prompt_gradual

            batch_responses = call_reasoner_batch(prompt, task_contents, gpt, 0.6)
            answers, rationales = zip(*[extract_final_answer_and_rationale(response, question_type) for response in batch_responses])

            verdicts = call_gate_batch(answers, rationales, task_contents, gate_model, gate_tokenizer, weighted_training, prompt_setting)
            if weighted_training:
                # Modified to include confidence scores
                confidence_0 = verdicts['confidence_0']  # confidence for 0
                confidence_1 = verdicts['confidence_1']  # confidence for 1
                verdicts = verdicts['predictions'] # temp solution to prevent bug - overwrite verdicts to match previous format
            
            remaining_task_indices = []
            for i, (answer, rationale, verdict) in enumerate(zip(answers, rationales, verdicts)):
                result_entry = {
                    'ID': ids[i],
                    'Turn': turn,
                    'Reasoner Task Content': task_contents[i],
                    'Reasoner Answer': normalize_answer(answer),
                    'Reasoner Rationale': rationale,
                    'Gold Answer': batch_gold_answers[i],
                    'Gate Output': verdict,
                    'Correct Answer': exact_match_multiple(answers[i], batch_gold_answers[i]),
                }

                # Add weighted training-specific fields if enabled
                if weighted_training:
                    result_entry['Confidence Scores'] = f"0: {confidence_0[i]:.4f}, 1: {confidence_1[i]:.4f}"

                # Append the entry to results
                results.append(result_entry)

                if verdict == 0:
                    remaining_task_indices.append(i)
            
            if (len(remaining_task_indices) == 0):
                break

            # Filter the tasks that still need to be processed
            task_contents = [task_contents[i] for i in remaining_task_indices]
            ids = [ids[i] for i in remaining_task_indices]
            answers = [answers[i] for i in remaining_task_indices]
            previous_answers = [previous_answers[i] for i in remaining_task_indices]
            rationales= [rationales[i] for i in remaining_task_indices]
            verdicts = [verdicts[i] for i in remaining_task_indices]
            batch_gold_answers = [batch_gold_answers[i] for i in remaining_task_indices]
            batch_responses = [batch_responses[i] for i in remaining_task_indices]

            for i, (answer, rationale) in enumerate(zip(answers, rationales)):
                answer = answer if answer is not None else ""
                rationale = rationale if rationale is not None else "" 

                if exact_match_multiple(answer, [previous_answers[i]]):
                    task_contents[i] = task_contents[i] + f"LLM Answer {turn}: " + answer + f"\nLLM Rationale {turn}: " + rationale + f"\n{exact_wrong_prompt}\n"
                else:
                    task_contents[i] = task_contents[i]  + f"LLM Answer {turn}: " + answer + f"\nLLM Rationale {turn}: " + rationale + f"\n{feedback_prompt}\n"
            
            previous_answers = answers
            turn += 1

        save_df = pd.DataFrame(results)
        save_results_to_csv(save_df, checkpoint_file)
        return results
    
    #Exploration prompt setting
    else:
        cur = 0
        results = []

        while cur <= max_turns:
            s=time.time()

            if cur == 0:
                if question_type == 'MCQ':
                    prompt = mcq_force_answer_prompt 
                elif question_type == 'MATH':
                    prompt = math_force_answer_prompt
                else:
                    prompt = force_answer_prompt 
                batch_responses = call_reasoner_batch(prompt, task_contents, gpt)
            else:
                if question_type == 'MCQ':
                    prompt = mcq_exploration_force_answer_prompt 
                elif question_type == 'MATH':
                    prompt = math_exploration_force_answer_prompt
                else:
                    prompt = exploration_force_answer_prompt
                batch_responses = call_reasoner_batch(prompt, task_contents, gpt, 0.6, 0.7)

            answers, rationales = zip(*[extract_final_answer_and_rationale(response, question_type) for response in batch_responses])

            verdicts = call_gate_batch(answers, rationales, task_contents, gate_model, gate_tokenizer, weighted_training, prompt_setting)
            
            if weighted_training:
                # Modified to include confidence scores
                confidence_0 = verdicts['confidence_0']  # confidence for 0
                confidence_1 = verdicts['confidence_1']  # confidence for 1
                verdicts = verdicts['predictions'] # temp solution to prevent bug - overwrite verdicts to match previous format


            remaining_task_indices = []
            for i, (answer, rationale, verdict) in enumerate(zip(answers, rationales, verdicts)):
                # Common fields for all results
                result_entry = {
                    'ID': ids[i],
                    'Turn': cur,
                    'Reasoner Task Content': task_contents[i],
                    'Reasoner Answer': normalize_answer(answer),
                    'Reasoner Rationale': rationale,
                    'Gold Answer': batch_gold_answers[i],
                    'Gate Output': verdict,
                    'Correct Answer': exact_match_multiple(answers[i], batch_gold_answers[i]),
                }

                # Add weighted training-specific fields if enabled
                if weighted_training:
                    result_entry['Confidence Scores'] = f"0: {confidence_0[i]:.4f}, 1: {confidence_1[i]:.4f}"

                # Append the entry to results
                results.append(result_entry)

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
                task_contents[i] = task_contents[i] + "Previous LLM Response: " + batch_responses[i] + "\nWrong answer! Try again.\n"

            cur += 1

        save_df = pd.DataFrame(results)
        save_results_to_csv(save_df, checkpoint_file)

        # Extract final answers in batch
        return results

def run_system(input_file, output_file, question_type, log_file, batch_size=8, max_turns=4, weighted_training=False, prompt_setting ='gradual',gpt=False):
    start_time = time.time()

    original_data = pd.read_csv(input_file)

    start_position = -1
    for i in range(start_position+1,len(original_data), batch_size): #len(validation_data)
        batch = original_data.iloc[i:i + batch_size]
        batch_ids = [entry['ID'] for _,entry in batch.iterrows()]
        batch_questions = [entry['Question'] for _,entry in batch.iterrows()]
        batch_gold_answers = [safe_literal_eval(entry['Answers']) for _,entry in batch.iterrows()]
        task_contents = [f"Q: {entry['Question']} \n" for _,entry in batch.iterrows()]

        batch_predicted_answers = main_batch(
            ids=batch_ids,
            task_contents=task_contents,
            question_type=question_type,
            batch_gold_answers=batch_gold_answers,
            checkpoint_file = output_file,
            max_turns = max_turns,
            weighted_training=weighted_training,
            prompt_setting=prompt_setting,
            gpt=gpt
        )
        batch_start = i
        print(f"Processed batch {batch_start // batch_size + 1}/{(len(original_data) + batch_size - 1) // batch_size}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Generation complete. Results saved to CSV.")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    df = pd.read_csv(output_file)
    results= custom_evaluation(df, "All Turns")
    last_entries = df.groupby('ID').tail(1)
    results2 = custom_evaluation(last_entries, "Final Answer/Turn")

    with open(log_file, "a") as f:
        f.write(f"\nCustom Evaluation Results for All Turns\n")
        for metric, value in results.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            elif isinstance(value, int):
                f.write(f"{metric}: {value}\n")
            elif value is None:
                f.write(f"{metric}: No valid entries\n")
            else:
                f.write(f"{metric}\n")
        write_divider(f)

    with open(log_file, "a") as f:
        f.write(f"\nCustom Evaluation Results for Final Answer/Turn\n")
        for metric, value in results2.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            elif isinstance(value, int):
                f.write(f"{metric}: {value}\n")
            elif value is None:
                f.write(f"{metric}: No valid entries\n")
            else:
                f.write(f"{metric}\n")
        write_divider(f)

    original_stdout = sys.stdout
    with open(log_file, "a") as f:
        sys.stdout = f
        try:
            sys.stdout = f
            f.write("Evaluation Per Turn\n")
            evaluate_per_turn(df, prompt_setting)
            get_formula_score(df.groupby('ID').head(1),zero_shot=True)
            write_divider(f)
            finally:
            sys.stdout = original_stdout

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Run the inference system with the DAI model with various parameters to get predictions on validation data.')
    
    # Define command-line arguments
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment, used for naming and paths.')
    parser.add_argument('--dm_path', type=str, help='Path/directory of the DAI model. Default is generated using the experiment name.')
    parser.add_argument('--input_path', type=str, required=False, help='Path to the input dataset CSV file. Default is generated using the experiment name.')
    parser.add_argument('--output_path', type=str, required=False, help='Path to the output dataset CSV file. Default is generated using the experiment name.')
    parser.add_argument('--log_path', type=str, required=False, help='Path to the log file. Default is generated using the experiment name.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size to process at a time.')
    parser.add_argument('--max_turns', type=int, help='Max turns of responses, starting at 0.')
    parser.add_argument('--question_type', type=str, choices=['MCQ', 'OEQ','MATH'], default='OEQ', help='Type of question: MCQ for multiple choice, OEQ for open-ended, MATH for open-ended math.')
    parser.add_argument('--weighted_training', action="store_true", default=False, help='Boolean flag to indicate whether to use weighted training.')
    parser.add_argument('--prompt_setting', type=str, choices=['gradual', 'exploration'], default='gradual', help='Type of prompt setting: "gradual" or "exploration".')
    parser.add_argument('--gpt', action="store_true", default=False, help='Use llama (default) or GPT if pass in True.')
    parser.add_argument('--gpt_model', type=str, required=False, help='GPT model to use, e.g., "gpt-4" or "gpt-3.5".')
    parser.add_argument('--llama_model', type=str, required=False, help='Path to the Llama model if GPT is not used.')

    # Parse the arguments
    args = parser.parse_args()

    # Set default for input_path based on experiment_name if not provided
    if args.input_path is None:
        args.input_path = f'data/original/{args.experiment_name}_dev.csv'
    if args.output_path is None:
        args.output_path = f'predictions/{args.experiment_name}_predictions.csv'
    if args.log_path is None:
        args.log_path = f'logs/{args.experiment_name}_log.txt'
    if args.dm_path is None:
        args.dm_path = f'dm_training/finetuned_checkpoints/{args.experiment_name}'

    # Dynamically set max_turns based on prompt_setting if not explicitly provided
    if args.max_turns is None:
        args.max_turns = 4 if args.prompt_setting == 'exploration' else 5

    # Initialize the accelerator (as before)
    accelerator = Accelerator()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    #print(torch.cuda.is_available())
    torch.cuda.empty_cache()

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


    ## GATE ###
    from transformers import AutoModelForSeq2SeqLM

    gate_model_id=args.dm_path
    gate_model_id_tok =args.dm_path + "_tokenizer"
    gate_tokenizer = AutoTokenizer.from_pretrained(gate_model_id_tok)
    gate_model = AutoModelForSeq2SeqLM.from_pretrained(
        gate_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Spread layers across GPUs
        )

    if args.gpt:
        gate_model, gate_tokenizer = accelerator.prepare(gate_model, gate_tokenizer)
    else:
        model, gate_model, tokenizer, gate_tokenizer = accelerator.prepare(model, gate_model, tokenizer, gate_tokenizer)

    # Call the run_system function with parsed arguments
    run_system(args.input_path, args.output_path, args.question_type, args.log_path, args.batch_size, args.max_turns, args.weighted_training, args.prompt_setting, args.gpt)