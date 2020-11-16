# Scattering Transforms for Audio in Python

Please cite:

```
@article{tfST,
	title={Three-Dimensional Fourier Scattering Transform and Classification of Hyperspectral Images},
	author={Ilya Kavalerov and Weilin Li and Wojciech Czaja and Rama Chellappa},
	journal={arXiv preprint arXiv:1906.06804},
	year={2019}
}
```

## Usage

See `scripts` directory.

### Creating data

```
scripts/wake/data.sh
```

### Training

```
scripts/wake_final/vulcan_baseline_train.sh
scripts/wake_final/vulcan_cortical_train.sh
```

### Eval to create DET curves

```
scripts/wake_final/vulcan_baseline_eval.sh
```

### Create universal adversarial noise attacks

```
scripts/wake_final/attack_univ_all.sh
```

### Create adversarial music attacks

To do this attack you will need to export a model of a different length first using:
```
scripts/wake/vulcan_export.124.sh
```

Then attack with:

```
scripts/wake_final/attack_univ_music.sh
```

### Eval adversarial music attacks

```
scripts/wake_final/attack_univ_music_eval.sh
```

## Installing

`conda env create -f venvtf1p15nb.yml`

## Versioning

Running on: Python 3.6.9, tensorflow 1.15, cuda/10.0.130, cudnn/v7.6.5.

For creating datasets, works with: ffmpeg/4.2.1, rubberband 1.8.2.
Also using vamp-plugin-sdk 2.9 2019-11-13, libsndfile Version 1.0.28, libsndfile Version 0.1.9.

## License

MIT
