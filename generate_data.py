from models.openModels import generate_text_gemma
from data.prompts import get_prompts
from dotenv import load_dotenv


def generate_dataset(save_path="data/gemma9b"):
    load_dotenv()

    prompts = get_prompts(sample_size=10, seed=42)
    outputs = generate_text_gemma(prompts, 
                            model_name="bartowski/gemma-2-9b-it-GGUF",
                            n=5,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=45,
                            max_tokens=8192,
                            device="cuda")
    
    del llm  
    destroy_model_parallel()
    gc.collect() 
    torch.cuda.empty_cache()  

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")
        break


if __name__ == "__main__":
    generate_dataset()
