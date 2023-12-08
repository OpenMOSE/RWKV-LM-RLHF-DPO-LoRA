# RWKV-LM-RLHF-DPO

This project aims to implement Direct Preference Optimization for RWKV. 

20231201: Original idea

# WARNING: Debugging, pre-release.

## Usages
1. DPO dataset:
   - Run `washingdpodata.ipynb`, which fetches data from `https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/`.
   - Modify the number of preference pairs you want to train on. Default to 1000.
2. General corpus dataset:
   - You might need a general corpus to maintain general performance. It's required as a parameter.
   - The size of the dataset might vary, but the larger the better.
   - Use `binidx`, at `https://github.com/Abel2076/json2binidx_tool`.
3. Run `train.py`:
   - Currently only RWKV-5 is supported.
   - Takes up much memory (15GB) for a relatively small model (0.4B). TODO: use LoRA to save memory.

The command:
