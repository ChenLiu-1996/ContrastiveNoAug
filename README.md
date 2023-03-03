# Contrastive Learning without Augmentations

Chen Liu (chen.liu.cl2482@yale.edu)



## Environement Setup
<details>
  <summary><b>On our Yale Vision Lab server</b></summary>

- There is a virtualenv ready to use, located at
`/media/home/chliu/.virtualenv/contrastive-no-aug/`.

- Alternatively, you can start from an existing environment "torch191-py38env",
and install the following packages:
```
python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python3 -m pip install wget gdown numpy matplotlib pyyaml click scipy yacs scikit-learn scikit-image
python3 -m pip install cython pot
```

If you see error messages such as `Failed to build CUDA kernels for bias_act.`, you can fix it with:
```
python3 -m pip install ninja
```

</details>
