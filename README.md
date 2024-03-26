# Hitchhiker's Guide to the Llama Training (on VACC)

Welcome to the Llama Training Guide! This document provides step-by-step instructions to set up and use the system for training Llama models using the LoRa method. Since VACC uses older Linux Kernels, it does not have adequate support for some key libraries used for training. This script evolved after a lot of trial and error, so use at your own caution.

## Prerequisites
- Access to at least 4 GPUs in a DeepGreen node, e.g.
```zsh
#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=60G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=jupyter
```
- Conda installed on your system.
- Access to the model weights (contact achawla1@uvm.edu or jstonge1@uvm.edu if needed).

## Setup Instructions

### Step 1: Create a Conda Environment
- Start by creating a new Conda environment with Python 3.9.
- Use the provided `requirements.txt` file to install the necessary packages. This may take a while
    ```bash
    conda env create -f environment.yml
    conda activate llama_env
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
    ```

### Step 2: Requisition Compute Resources
- Run the `start_jupyter.sh` script to requisition the compute resources with the necessary GPUs with a name of your job. Let the name of this job be `train_llama_jan_10`. You can change it anytime
    ```bash
    sh start-jupyter.sh --train_llama_jan_10
    ```
- Note: If you change the name of your new environment, you also need to modify it in the `.sbatch` file.

### Step 3: Accessing the Jupyter Notebook Server
- After running the script, a URL for the Jupyter notebook server will be generated. You can find this printed in the terminal or `train_llama_jan_10.out` file generated from the script.
- You might have to wait before the resources are granted to you.

### Step 4: Running the Notebook in VS Code or Local Machine
#### 4.1 VS-Code
- Open a remote connection in VS-Code to your VACC account.
- Create a Jupyer notebook wherever you want
- When you open to the notebook. Go to Select Kernel > Existing Jupyter Server > Paste the URL there.
#### 4.2 Local Machine
- Running it on local machine requires one additional step. You can forward the port from your login node to your local machine by running the following command on the Terminal of your local machine.
    ```bash
    ssh -NfL 8900:localhost:8900 vacc-user1.uvm.edu
    ```
- You can now use the same URL on your local machine to access the jupyter server.

### Step 5: Run the Jupyter Notebook
- Once set up, you can start running the Jupyter notebook.
- And follow the directions mentioned in the notebook comments

### Step 6: Setting Up the Model Weights
- In the notebook, change the directory to the model weights you wish to fine-tune.
- If you do not have the weights, reach out to @AC or @JSO.

### Step 7: Understanding LoRa Training
- The training method used is LoRa (Low-Rank Adaptation).
- Refer to the [original paper](https://arxiv.org/pdf/2106.09685.pdf) for detailed information.

### Step 8: Preparing Training Data
- Format your training data in JSON format.
- The structure should be an array of objects: `[{"input": ..., "output": ...}, ...]`.
- Save this data as `train.json`.

### Step 9: Configuring LoRa Parameters
- Customize the LoRa configuration by tuning `r` and `alpha` values as per your requirements.

### Step 10: Fine-Tuning Other Parameters
- You can further fine-tune your training by adjusting parameters like `learning rate`, `num_epochs`, etc.

## Conclusion
In an ideal world, you have now finally trained your first model. But there is a good chance and it did not work. If you that's the case, please feel free reach out to the designated @AC or @JSO (achawla1@uvm.edu, jstonge1@uvm.edu).
