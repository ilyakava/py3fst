# Scattering Transforms in Python

Please cite:

```
@article{tfST,
	title={Three-Dimensional Fourier Scattering Transform and Classification of Hyperspectral Images},
	author={Ilya Kavalerov and Weilin Li and Wojciech Czaja and Rama Chellappa},
	journal={arXiv preprint arXiv:TBA},
	year={2019}
}
```

### Usage

### Downloading Data

See [the GIC website](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and download for example the "corrected Indian Pines" and "Indian Pines groundtruth" datasets.


#### Create Custom Training/Testing Splits

One training/testing split is included. Create more by editing the variables `OUT_PATH`, `DATASET_PATH`, `ntrials`, and `datasetsamples` in `create_training_splits.m`, and running:

```
matlab -nodesktop -nosplash -r "create_training_splits"
```


## Versioning

Tested on Python 2.7.14 (Anaconda), tensorflow 1.10.1, cuda 9.0.176, cudnn-7.0 (8.0 might work too). Red Hat Enterprise Linux Workstation release 7.6 (Maipo). GeForce GTX TITAN X.

## License

MIT
