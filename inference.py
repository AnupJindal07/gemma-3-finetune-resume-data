from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import constants

MODEL_NAME = constants.trained_model_path


def load_model():
    """Load the fine-tuned Gemma-3-4b-it model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    return model, tokenizer


def analyze_resume(resume_text):
    """Generate AI-based resume analysis output."""
    model, tokenizer = load_model()

    # Ensure model is in evaluation mode
    model.eval()

    prompt = f"""
        Below is a resume. Analyze it and provide a structured response.

        ### Resume:
        {resume_text}

        ### Response:
        """

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")

    # output = model.generate(
    #     **inputs,
    #     max_length=512,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_p=0.9,
    #     num_return_sequences=1
    # )

    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=512,  # Adjust based on expected output size
            do_sample=True,  # Enable sampling for creative outputs
            temperature=0.7,  # Controls randomness
            top_p=0.9,  # Nucleus sampling for diverse results
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id  # Prevents generation issues
        )

    print(f"output {output}")

    # Decode output
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

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
