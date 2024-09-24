# CItruS
Official repository for the EMNLP 2024 paper [CItruS: Chunked Instruction-aware State Eviction for Long Sequence Modeling](https://arxiv.org/abs/2406.12018), by Yu Bai∗, Xiyuan Zou∗, Heyan Huang, Sanxing Chen, Marc-Antoine Rondeau, Yang Gao, and Jackie Chi Kit Cheung

## How to use
First set the environment:
```bash
pip install transformers==4.34.0 datasets sentencepiece
pip install accelerate bitsandbytes
pip install jieba fuzzywuzzy rouge
git clone https://github.com/ybai-nlp/CItruS
```

Next follow the sample use below:
```python
from CItruS.src.citrus_methods import generate_with_citrus
from transformers import AutoTokenizer, AutoModelForCausalLM

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
```

