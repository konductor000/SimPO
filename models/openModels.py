from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
import os
import gc
from dotenv import load_dotenv


def load_gguf(model_name, gguf):
    model = hf_hub_download(model_name, filename=gguf)

    return model


def generate_text_gemma(prompts, 
                        model_name="google/gemma-2-9b-it",
                        gguf=None,
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

    if gguf is not None:
        model_name = load_gguf(model_name, gguf)
    llm = LLM(model=model_name, gpu_memory_utilization=0.7)
    generated_texts = llm.generate(prompts, sampling_params)

    del llm  # Delete the model object
    gc.collect()  # Run garbage collection
    torch.cuda.empty_cache()  # Clear the CUDA cache

    generated_texts = [{"prompt": output.prompt, "outputs": \
        [answer.text for answer in output.outputs]} for output in generated_texts]

    return generated_texts

# Example usage
if __name__ == "__main__":
    prompts = [
        "Tell me a story of pidor? Pidor is a fictional antient Greek God that is very kind and strong. Pidor was known for his love to men and bravery etc."
    ] * 1
    outputs = generate_text_gemma(prompts, 
                        model_name="google/gemma-2-9b-it",
                        gguf=None,
                        n=5,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=45,
                        max_tokens=8192,
                        device="cuda")


    for output in outputs:
        print(len(output["outputs"]))
        print(output["prompt"])
        for answer in output["outputs"]:
            print([answer])
            print("-"*100)
