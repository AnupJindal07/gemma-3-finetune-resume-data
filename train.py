from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import constants

print("CUDA available:", torch.cuda.is_available())

model_name = constants.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as padding
tokenizer.padding_side = "right"  # Ensure padding is on the right for causal LM

# Apply 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)

# Enable training mode
model.config.use_cache = False
'''
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
'''
config = LoraConfig(
    task_type="CAUSAL_LM",          # Causal language modeling
    r=16,                                   # Rank of adaptation
    lora_alpha=32,                         # LoRA scaling parameter
    lora_dropout=0.1,                      # LoRA dropout
    target_modules=[                       # Target modules for Phi-3
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",                           # No bias training
    inference_mode=False                   # Training mode
)

model = get_peft_model(model, config)

dataset = load_dataset("json", data_files={"train": constants.process_file_path})

def tokenize_function(examples):
    input_texts = examples["prompt"]
    output_texts = examples["response"]
    
    # Format conversations using Gemma-3 IT chat template
    conversations = []
    for prompt, response in zip(input_texts, output_texts):
        # Create conversation in Gemma-3 IT format
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        conversations.append(conversation)
    
    # Use the tokenizer's chat template to format the conversations
    formatted_texts = []
    for conversation in conversations:
        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_texts.append(formatted_text)
    
    # Tokenize the formatted conversations
    tokenized = tokenizer(
        formatted_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=512,
        return_tensors=None
    )
    
    # Create labels by masking the user prompt tokens
    labels = []
    for i, conversation in enumerate(conversations):
        # Create the user part only to find where assistant response starts
        user_conversation = [{"role": "user", "content": conversation[0]["content"]}]
        user_text = tokenizer.apply_chat_template(
            user_conversation,
            tokenize=False,
            add_generation_prompt=True  # This adds the assistant prompt
        )
        
        # Tokenize user part to find the boundary
        user_tokens = tokenizer(user_text, add_special_tokens=False)["input_ids"]
        full_tokens = tokenized["input_ids"][i]
        
        # Create label sequence - start with all masked
        label_seq = [-100] * len(full_tokens)
        
        # Only learn from the assistant response part
        user_length = len(user_tokens)
        if user_length < len(full_tokens):
            # Copy the assistant response tokens to labels (unmask them)
            for j in range(user_length, len(full_tokens)):
                if full_tokens[j] != tokenizer.pad_token_id:  # Don't learn from padding
                    label_seq[j] = full_tokens[j]
        
        labels.append(label_seq)
    
    tokenized["labels"] = labels
    return tokenized

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
    return_tensors="pt"
)

training_args = TrainingArguments(
    output_dir=constants.model_checkpoint_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=50,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == "__main__":
    print(f"Started training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train()
    model.save_pretrained(constants.trained_model_path)
    tokenizer.save_pretrained(constants.trained_model_path)
    print(f"Completed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")