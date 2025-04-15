### Train Command 
``` 
CUDA_VISIBLE_DEVICES=6 python train.py /tiny-224 \
  --model vit_tiny_patch16_224 \
  --input-size 3 224 224 \
  --epochs 200 \
  --batch-size 128 \
  --lr 5e-4 \
  --workers 2 \
  --opt adamw \
  --weight-decay 0.05 \
  --sched cosine \
  --warmup-epochs 5 \
  --amp \
  --experiment vit-tiny-baseline-tiny224 \
  --output output/vit-tiny-baseline-tiny224


```