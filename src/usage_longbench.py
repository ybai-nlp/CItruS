# This file contains code applying Citrus to Longbench datasets
# Reference: https://github.com/THUDM/LongBench

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from citrus_methods import generate_with_citrus
import argparse
import random
from tqdm import tqdm
import numpy as np
import json
from longbench.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}
def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_longbench(datasets):
    dataset2prompt = json.load(open("longbench/dataset2prompt.json", "r"))
    ret_data = {}
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        prompt_format = dataset2prompt[dataset]
        data_all = [data_sample for data_sample in data]
        ret_data[dataset] = {"data": data_all, 'prompt_format': prompt_format}
    return ret_data

def process_longbench(test_example, tokenizer, dataset_name):
    if 'samsum' in dataset_name:
        source, query = '\n'.join(test_example.split("\n")[:-1]), '\n'.join(test_example.split("\n")[-1:])
    elif 'trivia' in dataset_name:
        source, query = '\n'.join(test_example.split("\n")[:-6]), '\n'.join(test_example.split("\n")[-6:])
    elif 'trec' in dataset_name:
        source, query = '\n'.join(test_example.split("\n")[:-2]), '\n'.join(test_example.split("\n")[-2:])
    elif 'qasper' in dataset_name:
        source, query = '\n\n'.join(test_example.split("\n\n")[:-3]), '\n\n'.join(test_example.split("\n\n")[-3:])
    elif "passage_retrieval" in dataset_name:
        source, query = '\n\n'.join(test_example.split("\n\n")[:-3]), '\n\n'.join(test_example.split("\n\n")[-3:])
        print("query = ", query)
    else:
        source, query = '\n\n'.join(test_example.split("\n\n")[:-2]), '\n\n'.join(test_example.split("\n\n")[-2:])
    return source, query

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "llama2" in model_name or 'mistral' in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def main(model_name, quantization_type, device_map, dataset_name, cache_type, chunk_size, k):
    device = torch.device("cuda") 
    seed_everything(42)
    datasets_load = load_longbench([dataset_name])
    dataset2maxlen = json.load(open("longbench/dataset2maxlen.json", "r"))
    
    if quantization_type=='none':
        model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, device_map=device_map, cache_dir='/network/scratch/x/xiyuan.zou/cache/transformers_cache')
    elif quantization_type=='4bit':
        model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, device_map=device_map, load_in_4bit=True, cache_dir='/network/scratch/x/xiyuan.zou/cache/transformers_cache')
    elif quantization_type=='8bit':
        model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, device_map=device_map, load_in_8bit=True, cache_dir='/network/scratch/x/xiyuan.zou/cache/transformers_cache')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nxtline_id = tokenizer.convert_tokens_to_ids('<0x0A>')
    
    state_eviction_config={
        "cache_type": cache_type, # support standard, instruction_aware_single, instruction_aware_dual
        "k": k, 
        "chunk_size": chunk_size
    }
    
    test_examples = datasets_load[dataset_name]['data']
    dataset_prompt_format = datasets_load[dataset_name]['prompt_format']
    max_gen_length = dataset2maxlen[dataset_name]
    system_summaries = []
    reference_summaries = []
    all_classes = []
    lengths = []
    end_token = nxtline_id if dataset_name == "samsum" else tokenizer.eos_token_id
    
    for test_example in tqdm(test_examples):
        test_example_str = dataset_prompt_format.format(**test_example)
        if dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            test_example_str = build_chat(tokenizer, test_example_str, model_name)
        prompt_context, prompt_instruction = process_longbench(test_example_str, tokenizer, dataset_name) 
        all_classes = test_example["all_classes"]
        lengths.append(test_example["length"])
        
        generation_config = {
            "max_new_tokens": max_gen_length,
            "eos_token_id": end_token,
            "num_beams": 1,
        }
        
        generated_text=generate_with_citrus(model, tokenizer, prompt_context, prompt_instruction, device, state_eviction_config, generation_config)
        final_answer=generated_text.strip()

        #evaluate answers
        print("final answer = ", final_answer)
        print("reference answer = ", test_example['answers'])
        system_summaries.append(final_answer)
        reference_summaries.append(test_example['answers'])
                
    result = scorer_e(dataset_name, system_summaries, reference_summaries, lengths, all_classes)
    
    print("Experimental Settings")
    print("Model name:", model_name)
    print("Dataset name:", dataset_name)
    print("Quantization:", quantization_type)
    print("Cache type", cache_type)
    print("Chunk size", chunk_size)
    print("K", k)
    print("-----------------------------------------")
    print("Experimental Results")
    print("results = ", result)
    

if __name__ == "__main__":
    inf=999999999
    parser = argparse.ArgumentParser()
    #model config
    parser.add_argument('--model_name', dest='model_name', action='store', required=False, default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--quantization_type', dest='quantization_type', action='store', required=False, default='none') #none, 4bit, 8bit
    parser.add_argument('--device_map', dest='device_map', action='store', required=False, default="auto")
    
    #data config
    parser.add_argument('--dataset_name', dest='dataset_name', action='store', required=False, default='qasper')
    
    #method config
    parser.add_argument("--cache_type", dest='cache_type', action='store', required=False, default='standard')
    parser.add_argument('--chunk_size', dest='chunk_size', action='store', required=False, default=256, type=int)
    parser.add_argument('--k', dest='k', action='store', required=False, default=768, type=int)
    
    args = parser.parse_args()
    args = vars(args)
    main(**args)