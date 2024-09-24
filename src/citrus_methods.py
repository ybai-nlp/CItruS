import torch
import math
from tqdm import tqdm
from convert_models.modify_llama import enable_llama_pos_shift_attention
from convert_models.modify_mistral import enable_mistral_pos_shift_attention

@torch.no_grad()
def decoding(model, tokenizer, past_key_values, pred_token_idx, generation_config):
    device=model.device
    
    outputs = model.generate(
        input_ids=pred_token_idx,
        attention_mask=torch.ones(past_key_values[0][0].size(-2)+1).unsqueeze(0),
        past_key_values=past_key_values,
        **generation_config
    )
    
    return outputs[0].detach().cpu()

@torch.no_grad()
def get_past_key_values_standard(input_ids_seg1, input_ids_seg2, model, k=None, cache_type=None, past_key_values=None):
    device=input_ids_seg2.device
    if input_ids_seg1!=None:
        input_ids=torch.cat([input_ids_seg1, input_ids_seg2], dim=-1)
    else:
        input_ids=input_ids_seg2
    
    past_key_values_org = past_key_values 
    outputs = model(input_ids, output_attentions=True, use_cache=True, past_key_values=past_key_values)
    past_key_values = outputs.past_key_values
    
    if cache_type=="all":
        selected_past_key_values = past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        return selected_past_key_values, pred_token_idx
    
    attention_from_context=outputs.attentions
    given_indices_for_each_layer=[]
    for attention_per_layer_from_context in attention_from_context:
        if past_key_values_org is None:
            att_weights_from_context=torch.mean(torch.mean(attention_per_layer_from_context[0,:,len(input_ids_seg1[0]):,:len(input_ids_seg1[0])], dim=0), dim=0)
            topk_att_weights, topk_indices=torch.topk(att_weights_from_context, k=min(k, att_weights_from_context.size(0)))
            topk_indices=torch.cat([topk_indices, torch.tensor(range(len(input_ids_seg1[0]), len(input_ids[0])), dtype=torch.long, device=device)]) 
        else:
            assert input_ids_seg1==None
            att_weights_from_context=torch.mean(torch.mean(attention_per_layer_from_context[0,:,:,:past_key_values_org[0][0].size(2)], dim=0), dim=0)
            topk_att_weights, topk_indices=torch.topk(att_weights_from_context, k=min(k, att_weights_from_context.size(0)))
            topk_indices=torch.cat([topk_indices, torch.tensor(range(len(past_key_values_org[0][0][0][0]),  len(past_key_values_org[0][0][0][0]) + len(input_ids[0])), dtype=torch.long, device=device)])
        topk_indices, _=torch.sort(topk_indices)
        given_indices_for_each_layer.append(topk_indices)
 
    selected_past_key_values = tuple(tuple(kv[:, :, given_indices_for_each_layer[layer_idx], :] for kv in layer) for layer_idx, layer in enumerate(past_key_values))
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    return selected_past_key_values, pred_token_idx

@torch.no_grad()
def get_past_key_values_instruction_aware_single(input_ids_seg1, input_ids_seg2, input_ids_query, model, k=None, past_key_values=None):
    device=input_ids_seg2.device
    if input_ids_seg1!=None:
        input_ids=torch.cat([input_ids_seg1, input_ids_seg2], dim=-1)
        input_ids_2=torch.cat([input_ids_seg1, input_ids_query], dim=-1)
    else:
        input_ids=input_ids_seg2
        input_ids_2=input_ids_query
    
    past_key_values_org = past_key_values 
    outputs = model(input_ids, output_attentions=True, use_cache=True, past_key_values=past_key_values)
    outputs_2 = model(input_ids_2, output_attentions=True, use_cache=True, past_key_values=past_key_values)
    past_key_values = outputs.past_key_values
    
    attention_from_query=outputs_2.attentions
    given_indices_for_each_layer=[]
    for attention_per_layer_from_query in attention_from_query:
        if past_key_values_org is None:
            att_weights_from_query=torch.mean(torch.mean(attention_per_layer_from_query[0,:,len(input_ids_seg1[0]):,:len(input_ids_seg1[0])], dim=0), dim=0)
            topk_att_weights, topk_indices=torch.topk(att_weights_from_query, k=min(k, att_weights_from_query.size(0)))
            topk_indices=torch.cat([topk_indices.long(), torch.tensor(range(len(input_ids_seg1[0]), len(input_ids[0])), dtype=torch.long, device=device)]) 
        else:
            assert input_ids_seg1==None
            att_weights_from_query=torch.mean(torch.mean(attention_per_layer_from_query[0,:,:,:past_key_values_org[0][0].size(2)], dim=0), dim=0)  
            topk_att_weights, topk_indices=torch.topk(att_weights_from_query, k=min(k, att_weights_from_query.size(0)))
            topk_indices=torch.cat([topk_indices, torch.tensor(range(len(past_key_values_org[0][0][0][0]),  len(past_key_values_org[0][0][0][0]) + len(input_ids[0])), dtype=torch.long, device=device)])
        topk_indices, _=torch.sort(topk_indices)
        given_indices_for_each_layer.append(topk_indices)
        
    selected_past_key_values = tuple(tuple(kv[:, :, given_indices_for_each_layer[layer_idx], :] for kv in layer) for layer_idx, layer in enumerate(past_key_values))
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    
    return selected_past_key_values, pred_token_idx

@torch.no_grad()
def get_past_key_values_with_instruction_aware_double(input_ids_seg1, input_ids_seg2, input_ids_query, model, k=None, past_key_values_context=None, past_key_values_query=None):
    device=input_ids_seg2.device
    if input_ids_seg1!=None:
        input_ids=torch.cat([input_ids_seg1, input_ids_seg2], dim=-1)
        input_ids_2=torch.cat([input_ids_seg1, input_ids_query], dim=-1)
    else:
        input_ids=input_ids_seg2
        input_ids_2=input_ids_query
    
    past_key_values_context_org = past_key_values_context
    past_key_values_query_org = past_key_values_query
    outputs = model(input_ids, output_attentions=True, use_cache=True, past_key_values=past_key_values_context)
    outputs_2 = model(input_ids_2, output_attentions=True, use_cache=True, past_key_values=past_key_values_query)
    past_key_values_context = outputs.past_key_values
    past_key_values_query = outputs_2.past_key_values
    
    attention_from_context=outputs.attentions
    attention_from_query=outputs_2.attentions
    given_indices_for_each_layer_for_context=[]
    given_indices_for_each_layer_for_query=[]
    for attention_per_layer_from_context, attention_per_layer_from_query in zip(attention_from_context, attention_from_query):
        if past_key_values_context_org is None: #kv pruning第一步
            att_weights_from_context=torch.mean(torch.mean(attention_per_layer_from_context[0,:,len(input_ids_seg1[0]):,:len(input_ids_seg1[0])], dim=0), dim=0)
            att_weights_from_query=torch.mean(torch.mean(attention_per_layer_from_query[0,:,len(input_ids_seg1[0]):,:len(input_ids_seg1[0])], dim=0), dim=0)
            topk_att_weights, topk_indices_for_context=torch.topk(att_weights_from_context, k=min(k, att_weights_from_context.size(0)))
            topk_att_weights, topk_indices_for_query=torch.topk(att_weights_from_query, k=min(k, att_weights_from_query.size(0)))
            topk_indices_for_context=torch.cat([topk_indices_for_context, torch.tensor(range(len(input_ids_seg1[0]), len(input_ids[0])), dtype=torch.long, device=device)]) 
        else:
            #kv pruning中间步
            att_weights_from_context=torch.mean(torch.mean(attention_per_layer_from_context[0,:,:,:past_key_values_context_org[0][0].size(2)], dim=0), dim=0)
            att_weights_from_query=torch.mean(torch.mean(attention_per_layer_from_query[0,:,:,:past_key_values_query_org[0][0].size(2)], dim=0), dim=0)  
            topk_att_weights, topk_indices_for_context=torch.topk(att_weights_from_context, k=min(k, att_weights_from_context.size(0)))
            topk_att_weights, topk_indices_for_query=torch.topk(att_weights_from_query, k=min(k, att_weights_from_query.size(0)))
            topk_indices_for_context=torch.cat([topk_indices_for_context, torch.tensor(range(len(past_key_values_context_org[0][0][0][0]),  len(past_key_values_context_org[0][0][0][0]) + len(input_ids[0])), dtype=torch.long, device=device)])
        
        topk_indices_for_context, _=torch.sort(topk_indices_for_context)
        topk_indices_for_query, _=torch.sort(topk_indices_for_query)
        given_indices_for_each_layer_for_context.append(topk_indices_for_context)
        given_indices_for_each_layer_for_query.append(topk_indices_for_query)
    
    selected_past_key_values_for_context = tuple(tuple(kv[:, :, given_indices_for_each_layer_for_context[layer_idx], :] for kv in layer) for layer_idx, layer in enumerate(past_key_values_context))
    selected_past_key_values_for_query = tuple(tuple(torch.cat([kv[:, :, given_indices_for_each_layer_for_query[layer_idx], :], selected_past_key_values_for_context[layer_idx][kv_idx][:,:,-input_ids_seg2.size(-1):,:]], dim=-2) for kv_idx, kv in enumerate(layer)) for layer_idx, layer in enumerate(past_key_values_query))
    pred_token_idx_from_context = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    pred_token_idx_from_query = outputs_2.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    
    return selected_past_key_values_for_context, selected_past_key_values_for_query, pred_token_idx_from_context, pred_token_idx_from_query


def generate_with_citrus(model, tokenizer, prompt_context, prompt_instruction, device, state_eviction_config, generation_config):
    k=state_eviction_config["k"]
    chunk_size=state_eviction_config["chunk_size"]
    
    input_ids_context = tokenizer(prompt_context, return_tensors="pt").input_ids
    input_ids_instruction = tokenizer(prompt_instruction, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids_context, input_ids_instruction=input_ids_context.to(device), input_ids_instruction.to(device)
    
    model.to(device)
    model.eval()
    #position shift
    if "llama" in model.config.model_type:
        enable_llama_pos_shift_attention(model)
    elif "mistral" in model.config.model_type:
        enable_mistral_pos_shift_attention(model)
    else:
        raise ValueError(f"{model.config.model_type} currently not supported")
    
    seq_len = input_ids_context.size(1)
    print(f"Input length: {seq_len}")
    print("--------Prefilling Start--------")
    num_segments=math.ceil(seq_len / chunk_size)
    pbar = tqdm(range(num_segments))
    
    #stage1: document understanding
    if num_segments<=1:
        selected_past_key_values, pred_token_idx=get_past_key_values_standard(None, input_ids_context, model, k, cache_type="all")
    else:
        if state_eviction_config["cache_type"]=="standard":
            for idx in pbar:
                if idx == 0:
                    input_ids_seg1 = input_ids_context[:, idx * chunk_size : min(seq_len, (idx + 1) * chunk_size)].to(device)
                    input_ids_seg2 = input_ids_context[:, (idx + 1) * chunk_size : min(seq_len, (idx + 2) * chunk_size)].to(device)
                    selected_past_key_values, pred_token_idx=get_past_key_values_standard(input_ids_seg1, input_ids_seg2, model, k)
                elif idx == 1:
                    continue
                else:
                    input_ids_seg2 = input_ids_context[:, idx * chunk_size : min(seq_len, (idx + 1) * chunk_size)].to(device)
                    selected_past_key_values, pred_token_idx=get_past_key_values_standard(None, input_ids_seg2, model, k, past_key_values=selected_past_key_values)
            context_past_key_values=selected_past_key_values
            
        elif state_eviction_config["cache_type"]=="instruction_aware_single":
            for idx in pbar:
                if idx == 0:
                    input_ids_seg1 = input_ids_context[:, idx * chunk_size : min(seq_len, (idx + 1) * chunk_size)].to(device)
                    input_ids_seg2 = input_ids_context[:, (idx + 1) * chunk_size : min(seq_len, (idx + 2) * chunk_size)].to(device)
                    selected_past_key_values, pred_token_idx=get_past_key_values_instruction_aware_single(input_ids_seg1, input_ids_seg2, input_ids_instruction, model, k)
                elif idx == 1:
                    continue
                else:
                    input_ids_seg2 = input_ids_context[:, idx * chunk_size : min(seq_len, (idx + 1) * chunk_size)].to(device)
                    selected_past_key_values, pred_token_idx=get_past_key_values_instruction_aware_single(None, input_ids_seg2, input_ids_instruction, model, k, past_key_values=selected_past_key_values)
            context_past_key_values=selected_past_key_values
        elif state_eviction_config["cache_type"]=="instruction_aware_double":
            for idx in pbar:
                if idx == 0:
                    input_ids_seg1 = input_ids_context[:, idx * chunk_size : min(seq_len, (idx + 1) * chunk_size)].to(device)
                    input_ids_seg2 = input_ids_context[:, (idx + 1) * chunk_size : min(seq_len, (idx + 2) * chunk_size)].to(device)
                    selected_past_key_values_for_context, selected_past_key_values_for_query, pred_token_idx_from_context, pred_token_idx_from_query=get_past_key_values_with_instruction_aware_double(input_ids_seg1, input_ids_seg2, input_ids_instruction, model, k)
                elif idx == 1:
                    continue
                else:
                    input_ids_seg2 = input_ids_context[:, idx * chunk_size : min(seq_len, (idx + 1) * chunk_size)].to(device)
                    selected_past_key_values_for_context, selected_past_key_values_for_query, pred_token_idx_from_context, pred_token_idx_from_query=get_past_key_values_with_instruction_aware_double(None, input_ids_seg2, input_ids_instruction, model, k, past_key_values_context=selected_past_key_values_for_context, past_key_values_query=selected_past_key_values_for_query)
            context_past_key_values=selected_past_key_values_for_query
        else:
            raise ValueError(f"{state_eviction_config['cache_type']} is not a supported cache type")
        
    #stage2: instruction following
    print("--------Instruction Following Start--------")
    selected_past_key_values, pred_token_idx = get_past_key_values_standard(None, input_ids_instruction, model, k, past_key_values=context_past_key_values)
    selected_past_key_values = tuple(tuple(kv[:, :, :-len(input_ids_instruction[0]), :] for kv in layer) for layer_idx, layer in enumerate(selected_past_key_values))
    selected_past_key_values, pred_token_idx = get_past_key_values_standard(None, input_ids_instruction, model, k, past_key_values=selected_past_key_values)
    print("KV size before generation:", selected_past_key_values[0][0].size(2))
    generated_ids=decoding(model, tokenizer, selected_past_key_values, pred_token_idx, generation_config)   
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return output_text