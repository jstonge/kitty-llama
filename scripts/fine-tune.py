"""
References for training values:
- https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da # good example of full blown training script
- https://www.philschmid.de/instruction-tune-llama-2
- https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy # really good article.

Lora specific-stuff
- https://lightning.ai/pages/community/lora-insights/
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

def main():

    base_model = f"/users/a/c/achawla1/main/Catalogue/LLM/Models/hugging_face/13Bf"
    output_dir = 'finetune_bba_fm'
    
    # switch to `device_map = "auto"` for multi-GPU
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map='auto')
    model.config.use_cache = False
    
    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Fix weird overflow issue with fp16 training
    tokenizer.padding_side = "right"

    dataset = load_dataset("jstonge1/bba_fm_data")
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"""### Intruction:
    Pretend you are an observant Assistant, a helpful bot that takes course catalogue data and returns a single clean JSON object. If available, the JSON objects should contain for each course in the text a "Number", "Title", "Description", "Prerequisite", and "Credits". There can be only one set of Course Numbers and multiple Prerequisites.  If the text is not a course description, return "[]".

    ### Input:
    {example['prompt'][i]}

    ### Output:
    {example['completion'][i]}
    """
            output_texts.append(text)
        
        return output_texts



    peft_params = LoraConfig(
                    r=128,  # the rank of the LoRA matrices
                    lora_alpha=16, #  the weight
                    lora_dropout=0.05,  # dropout to add to the LoRA layers
                    bias="none",  # add bias to the nn.Linear layers?
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA

                )


    training_params = TrainingArguments(
        output_dir="./logs",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        logging_steps=1,
        learning_rate=0.0003,
        weight_decay=0.001,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,   # if not -1,  it'll override num_train_epochs
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="constant"
    )

    trainer = SFTTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    peft_config=peft_params,
                    tokenizer=tokenizer,
                    args=training_params, # the parameters of the training: batch_size, report_to="wandb", max_steps, etc...
                    packing=False, #  this tells the trainer to pack sequences of `max_seq_lenght`
                    max_seq_length=2048,  # the desired input sequences, we could increase this even more
                    formatting_func=formatting_prompts_func  # The instruction template to apply to the examples
                )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Free memory for merging weights
    # From https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da
    # del model
    # torch.cuda.empty_cache()

    # from peft import AutoPeftModelForCausalLM

    # model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    # model = model.merge_and_unload()

    # output_merged_dir = os.path.join("finetune_bba_fm", "final_merged_checkpoint")
    # model.save_pretrained(output_merged_dir, safe_serialization=True)


if __name__ == "__main__":
    main()