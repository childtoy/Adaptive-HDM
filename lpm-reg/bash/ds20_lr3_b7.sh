
lr=1e-3
wd=0.1
block=7
ds=100

for zero in "m" 
do

name=reg-ds$ds-b$block-lr$lr-schdCS-wd$wd-zero$zero-normX-truelength

CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --name $name \
    --model unet \
    --ds $ds \
    --train \
    --feature 128 \
    --block $block \
    --channel '1,2,2,2,4,4,8' \
    --rate '1,1,2,1,2,1,2' \
    --lr $lr \
    --wd $wd \
    --zero_padding $zero \
    --wandb

done