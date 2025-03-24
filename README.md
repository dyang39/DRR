# DRR Experiment Setup and Usage

This repository provides a framework to run DRR experiments, evaluate models, and run baselines. Follow the steps below to set up and run your experiments.

## Prerequisites

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your/repository.git
   cd repository
   ```

2. Ensure you have all the required dependencies installed (refer to `requirements.txt` or installation instructions in the repo).

3. If you're using GPT, make sure to set your API key in the environment. You can do this by adding the following line to your .bashrc, .zshrc, or equivalent shell configuration file:

    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```

    Then, run:
     ```bash
    source ~/.bashrc  # or `source ~/.zshrc` for Zsh users
    ```

## Step 1: Create Datasets

To start, you'll need to create datasets. This can be done with the provided `create_datasets.sh` script.

4. Change the permissions of the `.sh` file to make it executable:
   ```bash
   chmod +x /bash_scripts/create_datasets.sh
   ```

2. Run the `create_datasets.sh` script:
   ```bash
   ./bash_scripts/create_datasets.sh
   ```

This script will generate the necessary dataset files to begin training.

## Step 2: Create DRR Experiment Script

To run the DRR experiment, you'll first need to create a custom script using `create_DRR.py`. This script will guide you through entering parameters and generating an executable `.sh` file for your experiment.

1. Run `create_DRR.py`:
   ```bash
   python create_DRR.py
   ```

2. Follow the prompts to enter details for the DRR experiment:
   - **Experiment Name**: Enter your custom experiment name, e.g., `DRRtest1`.
   - **Dataset Names**: Enter a comma-separated list of datasets (e.g., `commonsenseqa, openbookqa, piqa, winogrande`).
        - If you pass in multiple datasets, the training will be joint training for the DM.
        - If you pass in just one dataset, it will default to individual training for that dataset.
   - **GPT Usage**: Choose whether to use GPT by entering `yes` or `no`.
   - **Model Name or Path**: Specify the model name or path (e.g., `gpt-4o-mini-2024-07-18`).
   - **CUDA GPUs**: Provide the GPU devices to use, e.g., `1,2,3`.
   - **Optional Parameters**: Specify additional options like question type (`MCQ`, `OEQ`, `MATH`), prompt setting (`gradual` or `exploration`), and whether to use force rationale or weighted training.

   Example output after entering inputs:
   ```text
   Example usage:
      chmod +x bash_scripts/{script_filename}
      ./bash_scripts/{script_filename}
   ```
   Running and error commands saved to: `nohup_logs/DRRtest1_command_log.txt`
   Detailed evaluation statistics during generation and inference saved to: `logs/DRRtest1_commonsenseqa_log.txt` (and so on for each dataset)
   DM training information saved to: `logs/DRRtest1_log.txt`
   Predictions saved to: `predictions/DRRtest1_commonsenseqa_predictions.csv` (and so on for each dataset)

3. After you have entered all details, the script will generate a `.sh` file in the `bash_scripts` directory, which can be used to run the DRR experiment.

4. Change the permissions of the generated `.sh` file to make it executable:
   ```bash
   chmod +x bash_scripts/{script_filename}
   ```

5. Run the generated `.sh` file to start the experiment:
   ```bash
   ./bash_scripts/{script_filename}
   ```

## Step 3: Evaluating Predictions

Once the DRR experiment is complete, you can evaluate the predictions using `eval_DRR.py`.

1. The predictions for each dataset are saved in the `predictions/` directory in the format `{name}_{dataset_name}_predictions.csv`.

2. To evaluate the predictions, run `eval_DRR.py` with the predictions file:
   ```bash
   python eval.py {name}_{dataset_name}_predictions.csv
   ```

3. The script will output the accuracy and F1 score of the predictions.

   Example output:
   ```text
   ZS Accuracy: _
   ZS Formula Score: _
   DRR Accuracy for {name}_{dataset_name}: _
   DRR Formula Score: _
   ```

## Step 4: Run Baselines

You can also run baseline experiments using the `create_baselines.py` script.

1. Run `create_baselines.py`:
   ```bash
   python create_baselines.py
   ```

2. Follow the prompts to enter details for the baseline experiment:
   - **Dataset**: Enter the dataset name.
   - **Baseline Type**: Choose the type of baseline to run (e.g., `zero_shot_llama_abstain`, `self_critic`,etc.).

   Example output after entering inputs:
   ```text
   Example usage:
      chmod +x bash_scripts/{script_filename}
      ./bash_scripts/{script_filename}
   ```
   Running and error commands saved to: `nohup_logs/{dataset}_{baseline_type}_command_log.txt`
   Detailed midstep and evaluation statistics saved to: `logs/{dataset}_{baseline_type}_log.txt`
   Predictions saved to: `predictions/{dataset}_{baseline_type}_predictions.csv`

3. Change the permissions of the generated `.sh` file:
   ```bash
   chmod +x bash_scripts/{script_filename}
   ```

4. Run the generated `.sh` file to start the baseline experiment:
   ```bash
   ./bash_scripts/{script_filename}
   ```

5. Evaluation is in `logs/{dataset}_{baseline_type}.txt`

---

## Summary of Script Outputs

- **Experiment Logs**: `nohup_logs/{name}_command_log.txt` contains the running and error commands.
- **Evaluation Logs**: Detailed evaluation statistics saved to `logs/{name}_{dataset_name}.txt`.
- **Predictions**: Predictions are saved to `predictions/{name}_{dataset_name}_predictions.csv`.

## Troubleshooting

- If you encounter any issues with running the experiments, check the logs in the `nohup_logs/` directory for error messages.
- Make sure all paths and environment variables are correctly set up (e.g., `CUDA_VISIBLE_DEVICES` for GPU usage).

---

This `README.md` provides step-by-step instructions for downloading, setting up, running the DRR experiment, evaluating predictions, and running baselines.