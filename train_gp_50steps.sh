CUDA_VISIBLE_DEVICES=0 python3 -m train.train_GPmotion \
    --save_dir save/temp \
    --overwrite \
    --corr_noise \
    --dataset humanml \
    --eval_during_training \
    --diffusion_steps 50 \
    --corr_mode R_trsrot