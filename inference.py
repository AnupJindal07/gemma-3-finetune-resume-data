from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import constants
import os
import logging
from typing import List, Optional, Tuple 
import re

# Set environment variables for debugging CUDA issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

MODEL_NAME = constants.finetuned_model_name

logger = logging.getLogger(__name__)

# Global variables to cache model and tokenizer
_model = None
_tokenizer = None



def load_model(model_name: str, device: str):
    """Load the fine-tuned Gemma-3-4b-it model and tokenizer."""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    print("Loading model and tokenizer...")
    
    logger.info(f"Loading chat model: {model_name}")
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    
    # Load model
    load_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
    _model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    _model.eval()
    
    logger.info(f"Chat model loaded successfully")
    return _model, _tokenizer

class GemmaChatService:
    """Service for chat completions using Gemma model - loads once and reuses"""
    
    def __init__(self, 
                 model_name: str = constants.finetuned_model_name, 
                 device: Optional[str] = None,
                 max_new_tokens: int = 2048,
                 temperature: float = 0.2,
                 top_p: float = 0.95):
        """
        Initialize the Gemma chat service.
        
        Args:
            model_name: Name or path of the Gemma model on HuggingFace
            device: Device to run the model on ('cpu', 'cuda', 'mps', or None for auto-detection)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Load the model
        self._load_model()
    
    def _get_device(self, device: Optional[str]) -> str:
        """
        Determine the appropriate device for model inference.
        
        Args:
            device: Specified device or None for auto-detection
            
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if device:
            return device
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the Gemma model and tokenizer using cached loading function"""
        try:
            self.model, self.tokenizer = load_model(self.model_name, self.device)
            logger.info(f"Chat completion service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading chat model {self.model_name}: {str(e)}")
            raise
    
    def build_prompt_from_query(self, query: str, contexts: List[dict] = None) -> str:
        """
        Method 1: Generate a proper prompt from a query
        
        Args:
            query: User query/question
            contexts: Optional list of context documents
            
        Returns:
            Formatted prompt string
        """
        if contexts:
            ctx = "\n\n".join([
                f"[{c.get('meta', {}).get('source', 'Unknown')}] {c.get('doc', str(c))}" 
                for c in contexts
            ])
            prompt = (
                f"<system>You are a helpful assistant. Use the provided context to answer the user's question accurately.</system>\n"
                f"<user>Question: {query}\n\nContext:\n{ctx}</user>\n<assistant>"
            )
        else:
            prompt = f"<system>You are a helpful assistant.</system>\n<user>{query}</user>\n<assistant>"
        
        return prompt
    
    def build_prompt_with_system_context(self, system_context: str, user_query: str) -> str:
        """
        Method 1b: Generate a prompt with custom system context and user query
        
        Args:
            system_context: Custom system context/instructions
            user_query: User query/question
            
        Returns:
            Formatted prompt string
        """
        prompt = f"<system>{system_context}</system>\n<user>{user_query}</user>\n<assistant>"
        return prompt
    
    def generate_response_from_prompt(self, prompt: str, max_new_tokens: int=1024) -> str:
        """
        Method 2: Generate response from a given prompt
        
        Args:
            prompt: Complete formatted prompt
            
        Returns:
            Generated response text
        """
        if not prompt.strip():
            logger.warning("Empty prompt provided for completion")
            return "Sorry, I received an empty prompt."
        if max_new_tokens <= 0:
            logger.warning("Invalid max_new_tokens provided, using default value")
            max_new_tokens = self.max_new_tokens

        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt" #,
                #truncation=True,
                #max_length=max_length  # Limit input length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode and extract only the assistant's response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            match = re.search(r"<assistant>(.*?)</assistant>", full_response, re.DOTALL)
            if match:
                assistant_response = match.group(1).strip()
            else:
                assistant_response = full_response.split("<assistant>")[-1].strip()
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error while generating the response."
    
    def chat_complete(self, query: str, contexts: List[dict] = None) -> Tuple[str, str]:
        """
        Convenience method that combines both: generate prompt and response
        
        Args:
            query: User query
            contexts: Optional context documents
            
        Returns:
            Tuple of (prompt, response)
        """
        prompt = self.build_prompt_from_query(query, contexts)
        response = self.generate_response_from_prompt(prompt)
        return prompt, response
    
    def chat_completion(self, system_context: str, user_query: str, max_new_tokens: int=1024) -> Tuple[str, str]:
        """
        Convenience method that combines both: generate prompt and response
        
        Args:
            query: User query
            contexts: Optional context documents
            
        Returns:
            Tuple of (prompt, response)
        """
        prompt = self.build_prompt_with_system_context(system_context, user_query)
        response = self.generate_response_from_prompt(prompt)
        return prompt, response
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }


# Global instance for easy access
_chat_service = None 

def get_chat_service(model_name: Optional[str] = None, **kwargs) -> GemmaChatService:
    """
    Get or create a global chat service instance.
    
    Args:
        model_name: Name of the model to use (only used on first call)
        **kwargs: Additional arguments for GemmaChatService
        
    Returns:
        Global chat service instance
    """
 
    global _chat_service
    if _chat_service is None:
        _chat_service = GemmaChatService(
            model_name=model_name ,
            **kwargs
        )
    return _chat_service
     

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
    try:
        # Initialize the service
        chat_service = GemmaChatService(
            model_name=MODEL_NAME,
            device=None  # Auto-detect device
        )
        
        # Example usage
        test_query = "What is machine learning?"
        
        # Method 1: Build prompt then generate response
        prompt = chat_service.build_prompt_from_query(test_query)
        print(f"Generated Prompt:\n{prompt}\n")
        
        response = chat_service.generate_response_from_prompt(prompt, max_new_tokens=256)
        print(f"Generated Response:\n{response}\n")

        prompt2, response2 = chat_service.chat_completion("You are a helpful assistant. Analysis the input resume and provide feedback.", test_resume)
        print(f"Complete Chat Response:\n{response2}\n")

    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")

