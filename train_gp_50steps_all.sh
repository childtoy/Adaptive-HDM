CUDA_VISIBLE_DIVICES=0 python3 -m train.train_GPmotion \
    --save_dir save/all_noise_blend \
    --overwrite \
    --corr_noise \
    --dataset humanml \
    --eval_during_training \
    --diffusion_steps 50 \
    --corr_mode all \
    --noise_blend True \