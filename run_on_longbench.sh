module purge

# load environment
module load anaconda/3
conda activate citrus_env

cd ~/research/citrus/src
python usage_longbench.py --model_name=meta-llama/Llama-2-7b-chat-hf --cache_type=instruction_aware_double --chunk_size=256  --k=768