# CItruS
Official repository for the EMNLP 2024 paper [CItruS: Chunked Instruction-aware State Eviction for Long Sequence Modeling](https://arxiv.org/abs/2406.12018), by Yu Bai∗, Xiyuan Zou∗, Heyan Huang, Sanxing Chen, Marc-Antoine Rondeau, Yang Gao, and Jackie Chi Kit Cheung

<p align="center">
  <img src="figure_5.pdf" width="33%" height="33%">
</p>

## How to use
First set the environment:
```bash
conda create -y -n citrus_env python=3.9 cudatoolkit=11.3.1 --override-channels -c conda-forge -c nvidia
conda activate citrus_env
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

Run CItruS on LongBench datasets
```bash
bash run_on_longbench.sh --model_name=meta-llama/Llama-2-7b-chat-hf --dataset_name=qasper --cache_type=instruction_aware_single --chunk_size=256  --k=768 
```

## Citation
```
@misc{2406.12018,
Author = {Yu Bai and Xiyuan Zou and Heyan Huang and Sanxing Chen and Marc-Antoine Rondeau and Yang Gao and Jackie Chi Kit Cheung},
Title = {CItruS: Chunked Instruction-aware State Eviction for Long Sequence Modeling},
Year = {2024},
Eprint = {arXiv:2406.12018},
}
```
