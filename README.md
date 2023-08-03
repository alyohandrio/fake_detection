# Fake Detection
This repository provides interface to detect ai-generated images. Images' embeddings are firstly constructed using vision transformer, then linear layer is used to get logits and predict classes. The linear layer requires training.
## Data extraction
Script `loading.py` takes `n` images from pexels.com corresponding to `query` and saves them in `out_dir/0` directory. 
Usage `python3 loading.py --query=query --n=n --out=out_dir`.
Script `generation.py` generates images using descriptions from `prompts_file` file using Stable Diffusion and saves them in `out_dir/1` directory. 
Usage `python3 generation.py --prompts=prompts_file --out=out_dir`.
## Splitting data
Script `split.py` splits images from `data_dir` folder into three groups: train, test and val corresponding to `test_size` and `val_size`. Directory `data_dir` must have two subdirectories: 0, corresponding to real images and 1, corresponding to fake images. Resulting folders will have the same structure. 
Usage `python3 --data=data_dir --train=train_dir --test=test_dir --validation=validation_dir --test_size=test_size --val_size=val_size`.
## Feature extraction 
Script `feature_extraction.py` converts real and fake images from `images/0` and `images/1` to corresponding embeddings saved in `out/0` and `out/1` respectively using vision transformer. 
Usage `python3 feature_extraction.py --images=images --out=out`.
## Training
Script `training.py` trains linear head and saves it in desired location.
Usage `python3 training.py --features=features --out=out`.
## Prediction
Use function `predict_fakes(images_path, head_path="checkpoints/head.pth")` from `processing.py` to get predictions. The function returns tensor of predictions and corresponding filenames.

