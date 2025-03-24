import os

def generate_command_script(name, dataset_names, question_type, model_name, use_force_rationale=True, use_weighted_training=True, gpt=False, gradual_or_explore="gradual", gpus="1,2,3"):
    # Define each command with placeholders
    rationale_flag = "" if use_force_rationale else "--force_rationale"
    weighted_flag = "--weighted_training" if use_weighted_training else ""
    gpt_flag = "--gpt" if gpt else ""
    model_type = "--gpt_model" if gpt else "--llama_model"
    dataset_string = ", ".join(dataset_names)

    sh_content = f"""#!/bin/bash
# This script runs all commands with specified settings

mkdir -p nohup_logs

# Create a log file for commands and errors
log_file="nohup_logs/{name}_command_log.txt"

# Function to run a command and handle errors
run_command() {{
    local cmd="$1"
    echo "Running: $cmd" | tee -a "$log_file"  # Print the command being executed
    if ! bash -c "$cmd"; then
        echo "Error: Command failed - $cmd" | tee -a "$log_file"  # Log error to command log
        exit 1  # Exit immediately if a command fails
    fi
    echo "Finished: $cmd" | tee -a "$log_file"  # Print when the command has successfully finished
}}

# Running generation for each dataset
"""

    for dataset_name in dataset_names:
        sh_content += f"""run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 generation/generation.py --experiment_name {name}_{dataset_name} --input_path data/original/{dataset_name}_train.csv --question_type {question_type} {rationale_flag} --prompt_setting {gradual_or_explore} {gpt_flag} {model_type} {model_name}'\nrun_command 'CUDA_VISIBLE_DEVICES={gpus} python3 dm_training/prepare_training.py --experiment_name {name}_{dataset_name}'
"""

    sh_content += f"""
# Preparing datasets for training
run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 dm_training/join_data.py --experiment_name {name} --datasets {dataset_string}'

# Running main training
run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 dm_training/main.py --experiment_name {name} {weighted_flag}'

# Running inference for each dataset
"""

    for dataset_name in dataset_names:
        sh_content += f"""run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 inference/inference.py --experiment_name {name}_{dataset_name} --input_path data/original/{dataset_name}_dev.csv --dm_path dm_training/finetuned_checkpoints/{name} --question_type {question_type} {weighted_flag} --prompt_setting {gradual_or_explore} {gpt_flag} {model_type} {model_name}'
"""

    os.makedirs("./bash_scripts", exist_ok=True)
    # Write the content to a .sh file
    script_filename = f"run_{name}.sh"
    with open("bash_scripts/" + script_filename, "w") as file:
        file.write(sh_content)
    print(f"{script_filename} has been created in bash_scripts with the provided settings.")
    print(f'''Example usage:
    chmod +x bash_scripts/{script_filename}
    ./bash_scripts/{script_filename}''')
    #print(f"Running and error commands saved to: nohup_logs/{name}_command_log.txt")
    #print(f"Detailed evaluation statistics during generation and inference saved to: logs/{name}_{dataset_names[0]}.txt (and so on for each dataset)")
    #print(f"DM training information saved to: logs/{name}.txt")
    #print(f"Predictions saved to: predictions/{name}_{dataset_names[0]}_predictions.csv (and so on for each dataset)")

# Example usage
if __name__ == "__main__":
    # User-defined settings
    name = input("Enter your custom DRR experiment name i.e. 'DRRtest1': ")
    
    # Accepting multiple dataset names
    dataset_names_input = input("Enter the dataset names (comma-separated) (e.g., commonsenseqa, openbookqa, piqa, winogrande): ")
    dataset_names = [name.strip() for name in dataset_names_input.split(",")]

    gpt = input("Use GPT? (yes/no): ").strip().lower() == "yes"
    model_name = input("Enter GPT model name or path to Llama (e.g., gpt-4o-mini-2024-07-18 or /path/to/Meta-Llama-3.1-8B-Instruct/): ")
    gpus = input("Enter CUDA visible devices (e.g., 1,2,3): ")

    ignore = input("Optional parameters, press enter for the following inputs to use defaults: ")
    question_type = input("Enter the question type (MCQ, OEQ, MATH) (Default MCQ): ")
    gradual_or_explore = input("Use gradual or exploration prompt? (gradual/exploration) (Default exploration): ").strip().lower() 
    use_force_rationale = input("Use force rationale? (yes/no) (Default no): ").strip().lower() == "yes"
    use_weighted_training = input("Use weighted training? (yes/no) (Default no): ").strip().lower() == "yes"

    if question_type is None or question_type == "":
        question_type = "MCQ"
    if gradual_or_explore is None or gradual_or_explore == "":
        gradual_or_explore = "exploration"
    if use_force_rationale is None or use_force_rationale == "":
        use_force_rationale = False  # Default to False
    if use_weighted_training is None or use_weighted_training == "":
        use_weighted_training = False  # Default to False

    generate_command_script(name, dataset_names, question_type, model_name, use_force_rationale, use_weighted_training, gpt, gradual_or_explore, gpus)
