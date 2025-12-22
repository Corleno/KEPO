# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
pip install wandb
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

# upgrade trl
pip install --upgrade trl

# upgrade torch version
pip install --upgrade torch torchaudio torchvision

# vLLM support 
pip install vllm==0.11.2

# upgrade deepspeed version
pip install "deepspeed>=0.18.3"

# fix transformers version
# pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
pip install "transformers>=4.57.3"