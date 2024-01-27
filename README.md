![image](https://github.com/Triang-jyed-driung/RWKV-LM-RLHF-DPO/assets/62597758/27e12a22-bb75-4287-9dd3-84ca3a3e3809)![image](https://github.com/Triang-jyed-driung/RWKV-LM-RLHF-DPO/assets/62597758/efd266aa-af81-4c1d-ad45-4cbb6455571c)# RWKV-LM-RLHF-DPO

This project aims to implement Direct Preference Optimization for RWKV. 

20231201: Original idea

# WARNING: Debugging, pre-release.

## Usages
1. DPO dataset:
   - Run `washingdpodata.ipynb`, which fetches data from https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/
   - Modify the number of preference pairs you want to train on. Default to 1000.
   - It will generate 3 files, `trainset.save`, `validset.save` and `testset.save`. Note that validation and test sets are reserved for future use.
2. General corpus dataset / SFT dataset:
   - You might need a general corpus or SFT corpus to maintain general performance. It's required as a parameter.
   - The size of the dataset might vary, but the larger the better.
   - Use `binidx` at https://github.com/Abel2076/json2binidx_tool, but if you don't have one, use `default_text_document` in this repo.
3. Run `train.py`:
   - Currently only RWKV-5 is supported.
   - Takes up too much memory (24GB) for a relatively small model (0.4B). TODO: use LoRA to save memory.

My training command is provided as follows:
```
./RWKV-LM-RLHF-DPO/RWKV-v5/train.py --load_model ./RWKV-5-World-0.4B-v2-20231113-ctx4096.pth --wandb <WANDB> --proj_dir ./models_2 --ctx_len 4096 --epoch_count 4 --epoch_begin 0 --epoch_steps 2000 --data_file ./RWKV-LM/RWKV-v5/data/minipile --data_type binidx --vocab_size 65536 --epoch_save 1 --micro_bsz 1 --n_layer 24 --n_embd 1024 --pre_ffn 0 --head_qk 0 --lr_init 5e-6 --lr_final 1e-6 --warmup_steps 50 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 --accumulate_grad_batches 20 --enable_progress_bar True --ds_bucket_mb 200 --my_testing r3r4 --dpo 1 --dpo_train_file ./RWKV-LM-RLHF-DPO/trainset.save --dpo_general_corpus_ratio 0.8 --dpo_beta 0.02
```

I use a mixed loss of this form: 
```math
Loss = (dpo\_general\_corpus\_ratio) * Loss\_general + (1 - dpo\_general\_corpus\_ratio) * Loss\_DPO
```

If you set `dpo_general_corpus_ratio` to 0, it will do only DPO.

## Toy model

I uploaded a toy model:
https://huggingface.co/ZhangRC/RWKV-5-World-DPO-Alpha
This model is trained on approximately 10,000 DPO pairs for one epoch on solely English data.

AlignBench results (in Chinese)
| 模型名称 | 专业能力 | 中文理解 | 基本任务 | 数学计算 | 文本写作 | 综合问答 | 角色扮演 | 逻辑推理 | 中文推理 | 中文语言 | 总分 |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 原版 | 2.887 | 2.052 | 2.353 | 1.241 | 3.120 | 3.658 | 2.595 | 1.750 | 1.496 | 2.778 | 2.136 |
| DPO  | 3.048 | 2.500 | 2.632 | 1.348 | 3.467 | 4.763 | 3.517 | 1.924 | 1.636 | 3.321 | 2.479 |

These results show the model's cross-lingual transferablilty.


