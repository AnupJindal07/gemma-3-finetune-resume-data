from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import constants
import os

# Set environment variables for debugging CUDA issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

MODEL_NAME = constants.finetuned_model_name


# Global variables to cache model and tokenizer
_model = None
_tokenizer = None

def load_model():
    """Load the fine-tuned Gemma-3-4b-it model and tokenizer."""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    print("Loading model and tokenizer...")
    
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set pad token if not present (common issue with Gemma models)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True  # Add this for Gemma models
        )
        
        print(f"Model loaded successfully on device: {_model.device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to CPU if GPU loading fails
        print("Attempting to load on CPU...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
    return _model, _tokenizer


def analyze_resume(resume_text):
    """Generate AI-based resume analysis output."""
    model, tokenizer = load_model()

    # Ensure model is in evaluation mode
    model.eval()

    prompt = f"""Below is a resume. Analyze it and provide a structured response.

### Resume:
{resume_text}

### Response:"""

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Tokenize input with proper padding and attention mask
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True,
            add_special_tokens=True
        )
        
        print(f"Input token IDs shape: {inputs['input_ids'].shape}")
        print(f"Input token IDs: {inputs['input_ids']}")
        
        # Check for invalid token IDs
        if torch.any(inputs['input_ids'] < 0) or torch.any(inputs['input_ids'] >= tokenizer.vocab_size):
            print("Warning: Invalid token IDs detected!")
            # Clean invalid tokens
            inputs['input_ids'] = torch.clamp(inputs['input_ids'], 0, tokenizer.vocab_size - 1)
        
        # Move tensors to device safely
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            print("Starting generation...")
            
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=512,  # Reduced for safety
                do_sample=False,  # Start with greedy decoding for debugging
                temperature=1.0,  # Reset to default
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Attempting CPU fallback...")
        
        # Fallback to CPU
        try:
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to("cpu")
            
            # Move model to CPU for fallback
            if hasattr(model, 'cpu'):
                model = model.cpu()
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        except Exception as cpu_error:
            print(f"CPU fallback also failed: {cpu_error}")
            return "Error occurred during inference"

    print(f"Generated output shape: {output.shape}")
    print(f"Generated tokens: {output}")

    # Decode output
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated response_text: {response_text}")

    # Return only the generated content (remove the input prompt from output)
    return response_text.replace(prompt, "").strip()


# if __name__ == "__main__":
#     resume_text = sys.argv[1] if len(sys.argv) > 1 else "Sample resume text here."
#     print(analyze_resume(resume_text))

if __name__ == "__main__":
    # test_resume = """
    # Name: John Doe
    # Experience: 5 years in Software Engineering
    # Skills: Java, Spring Boot, Kubernetes
    # Education: BSc in Computer Science
    # """
    test_resume = """
    Name: Emily Davis
    Title: Software Engineer
    
    Professional Experience:
    A highly motivated Software Engineer with 9 years of experience in Software Engineering. Proficient in Docker, Spring Boot, Microservices, REST APIs, Python, Java. Previous work includes:
    - Implementing innovative solutions using Spring Boot.
    - Collaborating with cross-functional teams to optimize performance.
    - Mentoring junior developers and leading workshops.
    
    Education:
    BEng in Information Technology
    
    Additional Information:
    - Organized tech meetups and conferences.
    - Active contributor to GitHub projects.
    
    Technical Skills:
    Docker, Spring Boot, Microservices, REST APIs, Python, Java
    """

    result = analyze_resume(test_resume)
    print(result)
