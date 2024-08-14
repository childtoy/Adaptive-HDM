# Adaptive Hilbert Diffusion Models for Controllable Smoothness in Continuous Function Generation

**** It shold be changed ****
[![arXiv](https://img.shields.io/badge/arXiv-<2209.14916>-<COLOR>.svg)](https://arxiv.org/abs/2209.14916)

<!-- Implementation of **Adaptive-HDM** in Pytorch. It is a new approach to controlling smoothness in the generation of continuous functions using HDMs. It uses correlated noise according to length parameter by leveraging the underlying structures of Hilbert spaces and employing self-supervised smoothness estimation.  -->

This repo is the official Pytorch implementation of Adaptive-HDM: "[Adaptive Hilbert Diffusion Models for Controllable Smoothness in Continuous Function Generation]()".


**** It shold be changed ****
![Overview]()

## Environment <!--Installation-->

This code was tested on `Ubuntu 20.04.6 LTS` and requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

Setup conda env:
```shell
conda env create -f environment.yml
conda activate ahdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

## Dataset

### 1D synthetic data
To create 1D syntehtic data:
```shell
python3 -m lpm.dataset
```

After that, you can find the data for 1D function generation located at `./1d-generation/data`.
### Text-to-Motion
Fllow the instructions in [HumanML3D](), then locate at `./dataset/HumanML3D`.
```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

Download dependencies for Text-to-Motion:

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

## Pre-Trained model

- [Length Prediction Module]()
- [Text-to-motion]()

Unzip and place them in `./save/`.

## Training

### 1D Functional generation
```shell
cd 1d-generation
python3 -m main --train --corr
```

### Text-to-Motion
For Text to motion, we don't compute kernels during training, but use precomputed kernel values. To calculate these kernels, use the command below.

```shell
python3 -m train.pre_calK
```

Subsequently, you can train with below command.
```shell
python3 -m train.train_GPmotion --save_dir 'save/train' --corr_noise --dataset humanml --eval_during_training --diffusion_steps 50 --corr_mode R_trs --param_lenK_path HumanML3D_K_param_data196_fps20_dim263_len10.pkl
```

## Generation
### 1D Function generation
```shell
cd 1d-generation
python3 -m main --test --ckpt path/your/model
```

### Text-to-Motion
```shell
python3 -u sample.generate_GP --model_path path/your/model --num_samples 3 --num_repetitions 2 --dataset humanml --param_lenK_path HumanML3D_K_param_data196_fps20_dim263_len10.pkl --text_prompt "A man moves forward." --guidance_param 2.5 --corr_noise --corr_mode R_trs 
```

## Evaluation
### Text-to-Motion
```shell
python3 -m eval.eval_humanml -model_path path/your/model
```

## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi), [MDM](https://github.com/GuyTevet/motion-diffusion-model).

**** It shold be changed ****
## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.

**** It shold be changed ****
#### Bibtex
If you find this code useful in your research, please cite:

```
@inproceedings{
tevet2023human,
title={Human Motion Diffusion Model},
author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}
```