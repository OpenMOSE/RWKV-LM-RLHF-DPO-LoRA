# RWKV-LM-RLHF-DPO

This project aims to implement Direct Preference Optimization for RWKV. 

20231201: Original idea

# WARNING: Debugging, pre-release.

## Usages
1. DPO dataset:
   - Run `washingdpodata.ipynb`, which fetches data from `https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/`.
2. General corpus dataset:
   - You might need a general corpus to maintain general performance. It's highly recommended.
   - The size of the dataset might vary, but the larger the better.
   - Use `binidx`. 
3. Run `train.py`:
   - Currently only RWKV-5 is supported.
