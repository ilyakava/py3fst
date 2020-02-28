# Scattering Transforms in Python

Please cite:

```
@article{tfST,
	title={Three-Dimensional Fourier Scattering Transform and Classification of Hyperspectral Images},
	author={Ilya Kavalerov and Weilin Li and Wojciech Czaja and Rama Chellappa},
	journal={arXiv preprint arXiv:1906.06804},
	year={2019}
}
```

### Usage

The main file for HSI analysis is `hyper_pixelNN.py`.

To use this file run the following scripts:

- `sh scripts/final/train_dffn_IP_replicate.sh`, Replicates the results of DFFN for Indian Pines once.
- `sh scripts/final/train_EAP_PaviaU_replicate.sh`, Replicates the results of EAP-Area for PaviaU once.
- `python scripts/generate_many_dl_runs.py`, Generates scripts that will run 10 trials of DFFN and EAP for every dataset.
- `scripts/final/IP_fst_svm_all.sh`, runs FST and a SVM once every for every mask listed in a txt file.
- `scripts/final/svm_wst_all.sh`, runs WST and SVMs for the 4 datasets mentioned in the paper.
- `scripts/final/svm_wst_all.sh`, runs SVMs on the raw HSI images for the 4 datasets mentioned in the paper.`

### Downloading Data

You may download a copy of the HSI data mentioned in the paper [here](https://drive.google.com/file/d/1u6fzTztudcilKUmV9ZKUh6khZTIeAeB7/view?usp=sharing)

Or, see [the GIC website](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and download the datasets there.

### Training Masks

You may download the exact training masks we use in the paper [here](https://drive.google.com/file/d/1p3FB4VTHbLQJQPGzaG_jya5EgLM97qhi/view?usp=sharing) and place them in the `masks` directory.

When using options like `--svm_multi_mask_file_list` there should be a txt file that lists the fully qualified path to each mask file that should be used.

#### Create Custom Training/Testing Splits

Create more training/testing splits with `sites_train_val_split.py`.

## Versioning

Running on: Python 3.6.9, tensorflow 1.15, cuda/10.0.130, cudnn/v7.6.5.

Previously tested on Python 2.7.14 (Anaconda), tensorflow 1.10.1, cuda 9.0.176, cudnn-7.0 (8.0 might work too). Red Hat Enterprise Linux Workstation release 7.6 (Maipo). GeForce GTX TITAN X.

## License

MIT
