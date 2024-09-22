from models.openModels import generate_text_gemma
from models.reward import add_reward_to_samples
from data.prompts import get_prompts
from dotenv import load_dotenv
from datasets import Dataset
from config import Config

load_dotenv()


def generate_dataset(config):
    prompts = get_prompts(sample_size=config.sample_size, seed=42)

    outputs = generate_text_gemma(prompts=prompts, 
                            model_name=config.model_name,
                            gguf_filename=config.gguf_filename,
                            num_generations=config.num_generations,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            top_k=config.top_k,
                            max_tokens=config.max_tokens,
                            device=config.device,
                            gpu_memory_utilization=config.gpu_memory_utilization)

    outputs = add_reward_to_samples(samples=outputs, 
                                model_path=config.reward_model_path, 
                                device=config.device, 
                                batch_size=8)
    
    return outputs


if __name__ == "__main__":
    config = Config()
    data = generate_dataset(config)
    
    dataset = Dataset.from_list(data)
    dataset.push_to_hub("konductor/gemma2-SimPO-v1")
