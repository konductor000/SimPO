from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
import os
from dotenv import load_dotenv

def generate_text_gemma(prompts, 
                        model_name="google/gemma-2-9b-it",
                        n=1,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=45,
                        max_tokens=8192,
                        device="cuda"):
    """
    Load the Gemma 2 9B BF16 model and process input prompts using vLLM with batched inference.

    Args:
        prompts (list of str): List of input prompts to process.
        model_name (str): The name of the model to load from Hugging Face.
        device (str): The device to run the model on, e.g., "cuda" or "cpu".

    Returns:
        list of str: Generated text outputs corresponding to each input prompt.
    """
    # Load the tokenizer and model from Hugging Face
    # load_dotenv()
    # api_key = os.getenv("HF_TOKEN")

    # Initialize vLLM and generate outputs
    sampling_params = SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens)
    llm = LLM(model=model_name, gpu_memory_utilization=0.7)
    generated_texts = llm.generate(prompts, sampling_params)

    return generated_texts

# Example usage
if __name__ == "__main__":
    prompts = [
        "Tell me a story of pidor? Pidor is a fictional antient Greek God that is very kind and strong. Pidor was known for his love to men and bravery etc. 50 words"
    ]
    outputs = generate_text_gemma(prompts, model_name="google/gemma-2-9b-it")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")
