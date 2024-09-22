import os
import gc
from typing import List, Dict, Any

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Load environment variables from a .env file if present
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def load_gguf(model_name: str, gguf_filename: str) -> str:
    """
    Downloads the GGUF model file from Hugging Face Hub.

    Args:
        model_name (str): The name of the model repository.
        gguf_filename (str): The filename of the GGUF model.

    Returns:
        str: The local path to the downloaded GGUF model file.
    """
    try:
        model_path = hf_hub_download(repo_id=model_name, filename=gguf_filename, token=HF_TOKEN)
        return model_path
    except Exception as e:
        raise RuntimeError(f"Failed to download GGUF model: {e}")


def generate_text_gemma(
    prompts: List[str],
    model_name: str = "google/gemma-2-9b-it",
    gguf_filename: str = None,
    num_generations: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 45,
    max_tokens: int = 1024,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """
    Generates text outputs using the Gemma 2-9B model.

    Args:
        prompts (List[str]): A list of input prompts to process.
        model_name (str): The Hugging Face repository name of the model.
        gguf_filename (str, optional): The GGUF model filename. Defaults to None.
        num_generations (int): Number of text generations per prompt.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        top_k (int): Top-K sampling parameter.
        max_tokens (int): Maximum number of tokens to generate.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing prompts and their generated outputs.
    """
    # Load GGUF model if specified
    if gguf_filename:
        model_path = load_gguf(model_name, gguf_filename)
    else:
        model_path = model_name

    # Initialize sampling parameters
    sampling_params = SamplingParams(
        n=num_generations,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens
    )

    try:
        # Initialize the LLM with specified GPU memory utilization
        llm = LLM(model=model_path, gpu_memory_utilization=0.85, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {e}")

    try:
        # Generate texts for the provided prompts
        generated_texts = llm.generate(prompts, sampling_params)
    except Exception as e:
        raise RuntimeError(f"Text generation failed: {e}")
    finally:
        # Clean up resources
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # Format the generated outputs
    formatted_outputs = [
        {
            "prompt": output.prompt,
            "outputs": [answer.text for answer in output.outputs]
        }
        for output in generated_texts
    ]

    return formatted_outputs


if __name__ == "__main__":
    # Define a list of prompts
    prompts = [
        """Llama 2 is a collection of pretrained and fine-tuned generative text models ranging 
        in scale from 7 billion to 70 billion parameters. This is the repository for the 
        7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging 
        Face Transformers format. Links to other models can be found in the index at the bottom.

        Model Details
        Note: Use of this model is governed by the Meta license. In order to download the 
        model weights and tokenizer, please visit the website and accept our License before 
        requesting access here.

        Meta developed and publicly released the Llama 2 family of large language models (LLMs), 
        a collection of pretrained and fine-tuned generative text models ranging in scale from 
        7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama-2-Chat, are 
        optimized for dialogue use cases. Llama-2-Chat models outperform open-source chat models 
        on most benchmarks we tested, and in our human evaluations for helpfulness and safety, 
        are on par with some popular closed-source models like ChatGPT and PaLM.

        Model Developers Meta

        Variations Llama 2 comes in a range of parameter sizes — 7B, 13B, and 70B — as well as 
        pretrained and fine-tuned variations.

        Input Models input text only.

        Output Models generate text only.

        Model Architecture Llama 2 is an auto-regressive language model that uses an optimized 
        transformer architecture. The tuned versions use supervised fine-tuning (SFT) and 
        reinforcement learning with human feedback (RLHF) to align to human preferences for 
        helpfulness and safety.
        Llama 2 family of models. Token counts refer to pretraining data only. All models are trained 
        with a global batch-size of 4M tokens. Bigger models - 70B -- use Grouped-Query Attention 
        (GQA) for improved inference scalability.

        Model Dates Llama 2 was trained between January 2023 and July 2023.
        What is this model?
        """
    ] * 1  # Replicating the prompt 1000 times

    # Specify the model name
    model_name = "google/gemma-2-9b-it"

    # Generate text outputs
    outputs = generate_text_gemma(
        prompts=prompts,
        model_name=model_name,
        gguf_filename=None,
        num_generations=1,
        temperature=0.7,
        top_p=0.9,
        top_k=45,
        max_tokens=1024,
        device="cuda"
    )

    # Display the generated outputs
    for output in outputs:
        print(f"Number of Outputs: {len(output['outputs'])}")
        print(f"Prompt: {output['prompt'][:100]}...")  # Truncated for brevity
        for answer in output["outputs"]:
            print(f"Generated Text: {answer}")
            print("-" * 100)



