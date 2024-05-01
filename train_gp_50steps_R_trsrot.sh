python3 -m train.train_GPmotion \
    --save_dir save/R_trsrot_noise_blend \
    --overwrite \
    --corr_noise \
    --dataset humanml \
    --eval_during_training \
    --diffusion_steps 50 \
    --corr_mode R_trsrot \
    --noise_blend True \
    --device 0 