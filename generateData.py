from models.openModels import generate_text_gemma
from models.reward import add_reward_to_samples
from data.prompts import get_prompts
from dotenv import load_dotenv
from datasets import Dataset


def generate_dataset(save_path="data/gemma9b", model_name="google/gemma-2-9b-it", gguf=None):
    load_dotenv()
    prompts = get_prompts(sample_size=10000, seed=42)

    outputs = generate_text_gemma(prompts, 
                            model_name=model_name,
                            gguf=gguf,
                            n=5,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=45,
                            max_tokens=8192,
                            device="cuda")
    outputs = add_reward_to_samples(outputs, model_path="RLHFlow/ArmoRM-Llama3-8B-v0.1", device="cuda", batch_size=128)
    
    return outputs


if __name__ == "__main__":
    model_name="bartowski/gemma-2-9b-it-GGUF"
    gguf="gemma-2-9b-it-Q4_K_L.gguf"
    data = generate_dataset(save_path="data/gemma2_9b_4bit", 
                            model_name="google/gemma-2-9b-it", 
                            gguf=None)

    dataset = Dataset.from_dict(data)
    dataset.push_to_hub("konductor/gemma2-SimPO-v1")
