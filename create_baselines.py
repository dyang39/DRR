import os

def create_baselines(dataset, baseline_type, model_path, gpus="1,2,3"):
    # Define baseline types and corresponding flags
    if baseline_type == "self_critic":
        baseline_flag = ""
        command = f'CUDA_VISIBLE_DEVICES={gpus} python3 inference/self_critic.py --experiment_name {dataset}_{baseline_type} --input_path data/original/{dataset}_dev.csv --question_type MCQ'
    elif baseline_type == "zero_shot_llama":
        command = f'CUDA_VISIBLE_DEVICES={gpus} python3 inference/zero_shot.py --experiment_name {dataset}_{baseline_type} --input_path data/original/{dataset}_dev.csv --question_type MCQ --llama_model {model_path}'
    elif baseline_type == "zero_shot_gpt":
        command = f'CUDA_VISIBLE_DEVICES={gpus} python3 inference/zero_shot.py --experiment_name {dataset}_{baseline_type} --input_path data/original/{dataset}_dev.csv --question_type MCQ --gpt --gpt_model {model_path}'
    elif baseline_type == "zero_shot_abstain_llama":
        command = f'CUDA_VISIBLE_DEVICES={gpus} python3 inference/zero_shot.py --experiment_name {dataset}_{baseline_type} --input_path data/original/{dataset}_dev.csv --question_type MCQ --llama_model {model_path} --abstain'
    elif baseline_type == "zero_shot_abstain_gpt":
        command = f'CUDA_VISIBLE_DEVICES={gpus} python3 inference/zero_shot.py --experiment_name {dataset}_{baseline_type} --input_path data/original/{dataset}_dev.csv --question_type MCQ --gpt --gpt_model {model_path} --abstain'
    else:
        raise ValueError("Invalid baseline type provided.")

    # Creating the bash script content
    sh_content = f"""#!/bin/bash
# This script runs all commands with specified settings for {baseline_type} baseline

mkdir -p nohup_logs

# Create a log file for commands and errors
log_file="nohup_logs/{dataset}_{baseline_type}_command_log.txt"

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

# Running {baseline_type} baseline inference
run_command '{command}'
"""

    # Ensure the directory for bash scripts exists
    os.makedirs("./bash_scripts", exist_ok=True)

    # Write the content to a .sh file
    script_filename = f"run_{dataset}_{baseline_type}.sh"
    with open("bash_scripts/" + script_filename, "w") as file:
        file.write(sh_content)
    
    print(f"{script_filename} has been created in bash_scripts with the provided settings.")
    print(f'''Example usage:
    chmod +x bash_scripts/{script_filename}
    ./bash_scripts/{script_filename}''')
    #print(f"Running and error commands saved to: nohup_logs/{dataset}_{baseline_type}_command_log.txt")
    #print(f"Detailed midstep and evaluation statistics saved to: logs/{dataset}_{baseline_type}.txt")
    #print(f"Predictions saved to: predictions/{dataset}_{baseline_type}_predictions.csv")

# Example usage
if __name__ == "__main__":
    # User-defined settings
    dataset = input("Enter the dataset name: ").strip()
    baseline_type = input("Enter the type of baseline (self_critic, zero_shot_llama, zero_shot_gpt, zero_shot_abstain_llama, zero_shot_abstain_gpt): ").strip().lower()
    model_path = input("Enter the path to the model (either Llama or GPT): ").strip()
    gpus = input("Enter CUDA visible devices (e.g., 1,2,3): ").strip()

    create_baselines(dataset, baseline_type, model_path, gpus)
