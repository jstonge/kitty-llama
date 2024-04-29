import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

def main():
    # base_model = f"/users/a/c/achawla1/main/Catalogue/LLM/Models/hugging_face/13Bf"
    model = AutoModelForCausalLM.from_pretrained("finetune_bba_fm", device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("finetune_bba_fm", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("jstonge1/bba_fm_data", split="test")

    # Evaluate the new model
    eval_prompt = f"""### Intruction:
    Pretend you are an observant Assistant, a helpful bot that takes course catalogue data and returns clean JSON object. If available, the JSON objects should contain for each course in the text a "Number", "Title", "Description", "Prerequisite", and "Credits". There can be multiple Course Numbers and multiple Prerequisites, include all of them.  If the text is not a course description, return "[]".

    ### Input:
    {dataset['prompt'][10]}
    """

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")


    model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=1000)[0], skip_special_tokens=True))



if __name__ == "__main__":
    main()