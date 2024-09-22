from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
import os
import gc
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


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
                        max_tokens=1024,
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
    llm = LLM(model=model_name, gpu_memory_utilization=0.85)
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
        """Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. This is the repository for the 7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format. Links to other models can be found in the index at the bottom.

Model Details
Note: Use of this model is governed by the Meta license. In order to download the model weights and tokenizer, please visit the website and accept our License before requesting access here.

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama-2-Chat, are optimized for dialogue use cases. Llama-2-Chat models outperform open-source chat models on most benchmarks we tested, and in our human evaluations for helpfulness and safety, are on par with some popular closed-source models like ChatGPT and PaLM.

Model Developers Meta

Variations Llama 2 comes in a range of parameter sizes — 7B, 13B, and 70B — as well as pretrained and fine-tuned variations.

Input Models input text only.

Output Models generate text only.

Model Architecture Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety.
Llama 2 family of models. Token counts refer to pretraining data only. All models are trained with a global batch-size of 4M tokens. Bigger models - 70B -- use Grouped-Query Attention (GQA) for improved inference scalability.

Model Dates Llama 2 was trained between January 2023 and July 2023.
What is this model?
"""
    ] * 1000
    # prompts = ["Give me receipt of spagetti with onions and pig wings"] * 1000
    model_name = "google/gemma-2-9b-it"
    outputs = generate_text_gemma(prompts, 
                        model_name=model_name,
                        gguf=None,
                        n=1,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=45,
                        max_tokens=1024,
                        device="cuda")


    for output in outputs:
        print(len(output["outputs"]))
        print(output["prompt"])
        for answer in output["outputs"]:
            print([answer])
            print("-"*100)
