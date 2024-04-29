# Set up latest torch instance with Python=3.9
from pathlib import Path
import json
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainerCallback
from contextlib import nullcontext
from transformers import default_data_collator, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def create_peft_config(model):
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=256, # tune this
        lora_alpha=512, # and this
        lora_dropout=0.05,
        # the target modules can also be tuned
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


def main():
    
    root_dir = Path("..")
    model_id = root_dir / "weights" / "weights" / "13Bf_hf" # location for model directory, must be in torch format
    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16) # device_map='auto' will use GPU if available


    #  SYSTEM PROMPT AND TRAINING DATA ------------------------------------------------------------------------


    sys_prompt = """Pretend you are an observant Assistant, a helpful bot that takes course catalogue data and returns clean JSON object list. If available, the JSON objects should contain for each course in the text a "Number", "Title", "Description", "Prerequisite", and "Credits". There can be multiple Course Numbers and multiple Prerequisites, include all of them.  If the text is not a course description, return "[]"."""

    train_dataset = load_dataset('json', data_files = str(root_dir /'data' / 'training.json'), split='train')  

    # split train into train and validation
    train_dataset = train_dataset.train_test_split(test_size=0.09, shuffle=True)
    
    # train_dataset['train'][0]

    def formatting_func(example):
        text = f"""<s>[INST] <<SYS>> {sys_prompt} <</SYS>>

        ```{example['input']}``` 
        Make sure to include prerequisites and exclude any 
        non-course information. [/INST] {example['output']}
        """

        return text

    def generate_and_tokenize_prompt(prompt):
        return tokenizer(formatting_func(prompt)) # tokenize the data

    # formating train and validation set
    tokenized_train_dataset = train_dataset["train"].map(generate_and_tokenize_prompt)
    tokenized_validate_dataset = train_dataset["test"].map(generate_and_tokenize_prompt)

    tokenizer.pad_token = tokenizer.eos_token # llama quirk, have to do it
    model.gradient_checkpointing_enable() # makes the training faster

    model.train() # put the model in training mode


    # SET UP MODEL TRAINING WITH LORA CONFIG ------------------------------------------


    try:
        # create peft config
        model, lora_config = create_peft_config(model)
    except:
        model, lora_config = create_peft_config(model)

    enable_profiler = False
    output_dir = "tmp/llama-output"

    # also tune-able
    config = {
        'lora_config': lora_config,
        'learning_rate': 2.5e-5,
        'num_train_epochs': 2, # especially this one
        'gradient_accumulation_steps': 1,
        'per_device_train_batch_size': 1,
        'gradient_checkpointing': True,
    }

    # Set up profiler, and connect to wandb.ai
    if enable_profiler:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        
        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler
                
            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()


    # TRAIN THE MODEL, monitor on wandb ------------------------------------------


    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        fp16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy="steps",
        eval_steps=10,
        optim="adamw_torch_fused",
        max_steps=total_steps if enable_profiler else -1,
        **{k:v for k,v in config.items() if k != 'lora_config'}
    )

    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset= tokenized_validate_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[profiler_callback] if enable_profiler else [],
        )
        # Start training
        trainer.train()


    # WRAP-UP ------------------------------------------------------------------------


    # merge the weights, otherwise it will only save as lora weights, which llama.cpp does not support atm
    model = model.merge_and_unload()

    # save the model
    model.save_pretrained(root_dir / "outputs" / "13Bf_finetuned_02")
    tokenizer.save_pretrained(root_dir / "outputs" / "13Bf_finetuned_02")

if __name__ == "__main__":
    main()