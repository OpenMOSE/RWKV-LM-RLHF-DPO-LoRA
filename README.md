# RWKV-LM-RLHF-DPO

This project aims to implement Direct Preference Optimization for RWKV. 

20231201: Original idea

# WARNING: Debugging, pre-release.

## Usages
1. DPO dataset:
   - Run `washingdpodata.ipynb`, which fetches data from https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/
   - Modify the number of preference pairs you want to train on. Default to 1000.
   - It will generate 3 files, `trainset.save`, `validset.save` and `testset.save`. Note that validation and test sets are reserved for future use.
2. General corpus dataset:
   - You might need a general corpus to maintain general performance. It's required as a parameter.
   - The size of the dataset might vary, but the larger the better.
   - Use `binidx` at https://github.com/Abel2076/json2binidx_tool, but if you don't have one, use `default_text_document` in this repo.
3. Run `train.py`:
   - Currently only RWKV-5 is supported.
   - Takes up too much memory (15GB) for a relatively small model (0.4B). TODO: use LoRA to save memory.

The command is provided as follows, assuming that you are using the 0.4B model, and you don't have a general corpus dataset:
```
python "./RWKV-v5/train.py" --load_model "RWKV-5-World-0.4B-v2-20231113-ctx4096.pth" --proj_dir "./RWKV-v5" --data_file "default_text_document" --data_type binidx --vocab_size 65536 --ctx_len 4096 --epoch_steps 400 --epoch_count 20 --epoch_begin 0 --epoch_save 1 --micro_bsz 3 --n_layer 24 --n_embd 1024 --pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 5e-6 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 --my_testing r3r4 --dpo 1 --dpo_train_file "./trainset.save" --dpo_general_corpus_ratio 0 --dpo_beta 0.01
```

I use a mixed loss of this form: $$ Loss = (dpo\_general\_corpus\_ratio) * Loss\_general + (1 - dpo\_general\_corpus\_ratio) * Loss\_DPO $$

If you set `dpo_general_corpus_ratio` to 0, it will do only DPO.
