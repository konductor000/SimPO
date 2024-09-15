import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from time import perf_counter

def add_reward_to_samples(samples, model_path="RLHFlow/ArmoRM-Llama3-8B-v0.1", device="cuda", batch_size=16):
    # Initialize the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Process samples in batches
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]

        # Prepare batch inputs
        messages = [
            [{"role": "user", "content": sample["prompt"]}, {"role": "assistant", "content": output}]
            for sample in batch_samples
            for output in sample["outputs"]
        ]

        # Tokenize the inputs
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True, truncation=True).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids)
            preference_scores = outputs.score.cpu().float()

        # Add single preference score to each sample
        index = 0
        for sample in batch_samples:
            sample_rewards = []
            for _ in sample["outputs"]:
                reward_data = preference_scores[index].item()
                sample_rewards.append(reward_data)
                index += 1
            sample["rewards"] = sample_rewards

# Example usage
if __name__ == "__main__":
    samples = [
        {"prompt": "What are some synonyms for the word 'beautiful'?", "outputs": ["Gorgeous", "Stunning", "Nice", "Awefull", "Dick"]},
    ]

    start = perf_counter()
    add_single_reward_to_samples(samples, batch_size=128)
    end = perf_counter()
    print(end - start)

    for i, sample in enumerate(samples):
        print(sample)
        if i >= 4:
            break