import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from time import perf_counter
from typing import List, Dict, Any

def load_model_and_tokenizer(model_path: str, device: str) -> (AutoModelForSequenceClassification, AutoTokenizer):
    """
    Loads the pre-trained model and tokenizer.

    Args:
        model_path (str): Path or identifier of the pre-trained model.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        model: The loaded sequence classification model.
        tokenizer: The loaded tokenizer.
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model or tokenizer: {e}")

def prepare_batch_inputs(samples: List[Dict[str, Any]], tokenizer: AutoTokenizer, device: str) -> torch.Tensor:
    """
    Prepares the input tensors for the model from the samples.

    Args:
        samples (List[Dict[str, Any]]): List of samples containing prompts and outputs.
        tokenizer (AutoTokenizer): The tokenizer to encode the inputs.
        device (str): Device to move the tensors to.

    Returns:
        torch.Tensor: Tokenized input IDs.
    """
    messages = [
        [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": output}
        ]
        for sample in samples
        for output in sample["outputs"]
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        return inputs
    except Exception as e:
        raise RuntimeError(f"Error during tokenization: {e}")

def add_rewards_to_samples(
    samples: List[Dict[str, Any]],
    model,
    tokenizer,
    device: str,
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Adds reward scores to each output in the samples.

    Args:
        samples (List[Dict[str, Any]]): Samples with prompts and outputs.
        model: The sequence classification model.
        tokenizer: The tokenizer for encoding inputs.
        device (str): Device to perform computations on.
        batch_size (int, optional): Number of samples per batch. Defaults to 16.

    Returns:
        List[Dict[str, Any]]: Samples with added reward scores.
    """
    samples = samples.copy()

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]

        try:
            input_ids = prepare_batch_inputs(batch_samples, tokenizer, device)
        except RuntimeError as e:
            print(e)
            continue
        
        with torch.no_grad():
            try:
                outputs = model(input_ids=input_ids)
                preference_scores = outputs.score.cpu().float()
            except Exception as e:
                print(f"Error during model inference: {e}")
                continue

        index = 0
        for sample in batch_samples:
            rewards = [float(preference_scores[index + j]) for j in range(len(sample["outputs"]))]
            sample["rewards"] = rewards
            index += len(sample["outputs"])

    return samples

def add_reward_to_samples(
    samples: List[Dict[str, Any]],
    model_path: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    device: str = "cuda",
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Main function to add rewards to samples.

    Args:
        samples (List[Dict[str, Any]]): Input samples with prompts and outputs.
        model_path (str, optional): Path to the reward model. Defaults to "RLHFlow/ArmoRM-Llama3-8B-v0.1".
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to "cuda".
        batch_size (int, optional): Number of samples per batch. Defaults to 16.

    Returns:
        List[Dict[str, Any]]: Samples enriched with reward scores.
    """
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    samples_with_rewards = add_rewards_to_samples(samples, model, tokenizer, device, batch_size)
    return samples_with_rewards

def main():
    """
    Example usage of the add_reward_to_samples function.
    """
    samples = [
        {
            "prompt": "What are some synonyms for the word 'beautiful'?",
            "outputs": ["Gorgeous", "Stunning", "Nice", "Awefull", "Dick"]
        },
    ] * 2

    start_time = perf_counter()
    try:
        enriched_samples = add_reward_to_samples(samples, batch_size=2)
    except RuntimeError as e:
        print(e)
        return
    end_time = perf_counter()

    print(f"Processing Time: {end_time - start_time:.2f} seconds\n")

    for sample in enriched_samples:
        print(sample)

if __name__ == "__main__":
    main()