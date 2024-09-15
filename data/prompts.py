from datasets import load_dataset
import random

def get_random_prompts(sample_size=10000, seed=42):
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    prompts = [entry['prompt'] for entry in dataset]

    random.seed(seed)
    random_sample = random.sample(prompts, sample_size)

    return random_sample

# Example usage
if __name__ == "__main__":
    seed = 42
    sample_size = 10
    random_prompts = get_random_prompts(seed, sample_size)
    print(random_prompts)
