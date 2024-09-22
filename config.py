from dataclasses import dataclass

@dataclass
class Config:
    save_path: str = "data/gemma9b"
    model_name: str = "google/gemma-2-9b-it"
    gpu_memory_utilization = 0.88
    gguf_filename: str = None  
    sample_size: int = 10000
    seed: int = 42
    num_generations: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 45
    max_tokens: int = 8192
    device: str = "cuda"
    reward_model_path: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
