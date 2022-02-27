## Single node multiple GPU launch script
- ```CUDA_VISIBLE_DEVICES```中GPU数量要与```nproc_per_node```对应

### Pre-training
- ```OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,2,3,5 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
    --accum_iter 1 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --batch_size 32
  ```

### Fine-tuning
- ```OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,2,3,5 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune './offical_pt_weights/mae_pretrain_vit_base.pth' \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path '../datasets/flower_data/' --nb_classes 5
```