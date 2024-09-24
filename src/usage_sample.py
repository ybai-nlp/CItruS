from citrus_methods import generate_with_citrus
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    tokenizer = AutoTokenizer.from_pretrained("your_model")
    model = AutoModelForCausalLM.from_pretrained("your_model") # currently support Llama2, Llama3 and Mistral
    device = "Your device"
    
    prompt_context = "Enter your context here"
    prompt_instruction = "Enter your instruction here"
    
    state_eviction_config={
        "cache_type": "Specify which state eviction method you want to apply during prefilling", # support standard, instruction_aware_single, instruction_aware_dual
        "k": 768, 
        "chunk_size": 256
    }
    
    generation_config = {
        "max_new_tokens": 20,
        "do_sample": False,
        "num_beams": 1,
    }
    
    generated_text=generate_with_citrus(model, tokenizer, prompt_context, prompt_instruction, device, state_eviction_config, generation_config)
    print(generated_text)
    
    
if __name__ == "__main__":
    main()