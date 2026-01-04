from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    dtype=torch.float16,
    quantization_config=quantization_config,
)

# Enable training mode
model.config.use_cache = False

# Prepare model for k-bit training BEFORE applying LoRA
model = prepare_model_for_kbit_training(model)

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
    target_modules=[                       # Target modules for Gemma-3
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",                           # No bias training
    inference_mode=False                   # Training mode
)

model = get_peft_model(model, config)

# Print LoRA info to verify it's working
model.print_trainable_parameters()

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
        max_length=2048,
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
    
    # Debug: print first example
    if len(labels) > 0:
        non_masked_count = sum(1 for x in labels[0] if x != -100)
        print(f"Debug: First example has {non_masked_count} non-masked tokens to learn from")
    
    tokenized["labels"] = labels
    return tokenized

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove the original text columns that are causing issues
tokenized_datasets = tokenized_datasets.remove_columns(["prompt", "response"])

# Split into train and eval datasets (85-15 ratio)
train_eval_split = tokenized_datasets["train"].train_test_split(test_size=0.20, seed=42)
train_dataset = train_eval_split["train"]
eval_dataset = train_eval_split["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")
batch_size = 2
gradient_accumulation_steps = 32
num_epochs = 3


# Don't use DataCollatorForLanguageModeling as it interferes with our custom labels
# Use a simple data collator that just handles padding
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)

training_args = TrainingArguments(
    output_dir=constants.model_checkpoint_path,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*2,
    gradient_accumulation_steps=gradient_accumulation_steps, 
    num_train_epochs=num_epochs,
    learning_rate=2e-4,  # Higher learning rate for LoRA
    weight_decay=0.01,
    warmup_steps=20,
    
    logging_dir="logs",
    logging_steps=20,

    eval_steps=20,
    eval_strategy="steps",
    save_steps=20,
    save_total_limit=4,
    load_best_model_at_end=True,
    max_grad_norm=1.0,   # Gradient clipping
    #metric_for_best_model="eval_loss",
    greater_is_better=False, 
    fp16=False,                      # Use bfloat16 instead
    bf16=True,                       # Better for modern GPUs
    dataloader_pin_memory=False,     # Save memory, Reduce memory pressure
    gradient_checkpointing=True,     # Trade compute for memory
    remove_unused_columns=False,     # Keep all columns including labels
    report_to=None,                  # Disable logging to avoid conflicts
    run_name=f"gemma-3-4b-resume-qlora-{batch_size}bs-{num_epochs}ep",
    seed=42,
    #torch_compile=False,             # Disable torch compile for compatibility
    #dataloader_num_workers=0,        # Single-threaded data loading for stability
)

from transformers import EarlyStoppingCallback
# Configure a callback to stop training if the evaluation loss
# doesn't improve for 3 consecutive evaluations.
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping]  
)



if __name__ == "__main__":
    print(f"Started training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train()
    model.save_pretrained(constants.trained_model_path)
    tokenizer.save_pretrained(constants.trained_model_path)
    print(f"Completed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")