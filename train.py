from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import constants

print("CUDA available:", torch.cuda.is_available())

model_name = constants.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as padding

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
    task_type=TaskType.CAUSAL_LM,          # Causal language modeling
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

    inputs = tokenizer(input_texts, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(output_texts, padding="max_length", truncation=True, max_length=512)

    inputs["labels"] = labels["input_ids"]  # Add labels for loss computation
    return inputs

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

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
    data_collator=None,
)

if __name__ == "__main__":
    print(f"Started training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train()
    model.save_pretrained(constants.trained_model_path)
    tokenizer.save_pretrained(constants.trained_model_path)
    print(f"Completed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")