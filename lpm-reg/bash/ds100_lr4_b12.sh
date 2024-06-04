
lr=1e-4
wd=0.1
block=12
ds=100

for zero in "full" "m"
do

name=reg-ds$ds-b$block-lr$lr-schdCS-wd$wd-zero$zero-normX

CUDA_VISIBLE_DEVICES=7 python -u main.py \
    --name $name \
    --model unet \
    --ds $ds \
    --train \
    --feature 128 \
    --block $block \
    --channel '1,1,1,2,2,2,4,4,4,8,8,8' \
    --rate '1,1,1,1,1,1,2,1,1,2,1,2' \
    --lr $lr \
    --wd $wd \
    --zero_padding $zero \
    --wandb

done