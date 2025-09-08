# DiffLI2D: Unleashing the Potential of the Semantic Latent Space in Diffusion Models for Image Dehazing (ECCV2024)

## Introduction
This project implements an image dehazing algorithm (**DiffLI2D**) based on features from diffusion models (DDPM). By extracting features from different timesteps of the diffusion model and combining attention mechanisms with feature enhancement modules, the method restores clarity to hazy images. The project includes full training, validation and testing pipelines.


## Environment Setup
    conda env create -f environment.yaml
    conda activate difflid


## Dataset
Please download the Dense_Haze_NTIRE19, NH-Haze, and SOTS datasets, and organize them as follows respectively.

```
dataset/
├── train/
│   ├── hazy/    
│   └── clean/     
├── val/
│   ├── hazy/
├── └── clean/
```
Note: During model training or testing, you can directly modify the directory parameters (`--hazy_dir`,`--clean_dir`,`--val_hazy_dir`,`--val_clean_dir`) in the `training.py` file and pass in the exact path (applicable to the Dense_Haze_NTIRE19 and NH-Haze datasets). Alternatively, you can execute the `generate-path.py` file in the `data` directory, pass in the paths of haze images and clean images to generate a paired txt file, then pass the path of this txt file to the `--hazy_dir` and `--clean_dir` parameters in the `training.py` file (applicable to the SOTS datasets). The same applies to the validation set.

Generate a paired txt file：
```bash
python generate_path.py \
    --gt_dir /path/to/clean_images \
    --haze_dir /path/to/hazy_images \
    --output SOTS-pairs.txt
```

The format of the generated paired txt file is as follows:
```
haze_image_path|clean_image_path
```



## Training

### Prepare pre-trained model
1. Prepare the pre-trained DDPM model weight file([256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt),[256x256_diffusion_uncond](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)), which can be obtained from [the official guided-diffusion repository](https://github.com/openai/guided-diffusion). 
2. Place them under the `checkpoint` path.
3. Modify the `model_path` and `classifier_path` parameters in the `training.py` file.

### Pipeline
1. Pass the environment variable. And add parameters such as `--work_dir`, `--hazy_dir`, `--clean_dir`, `--val_hazy_dir`, `--val_clean_dir` to `MODEL_FLAGS`. 
```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
```

2. Enable distributed training. Modify `--nnodes` and `--nproc_per_node`.
```bash
torchrun --nnodes=1 --nproc_per_node=2 --standalone  training.py $MODEL_FLAGS
```


## Testing

Use the trained model for testing.

1. Add the `resume_from` parameter to `model_flags` to pass in the path of the pre-trained model.
2. Comment out line 114 `trainer.train(train_sampler)` in the `training.py` file, and use `# trainer.test()` on line 115 for testing.
3. The test command line is the same as that of the training phase.


If you have any time, please email hj0117@mail.ustc.edu.cn
