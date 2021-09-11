# MCML (Multi-view co-similarity metric learning)

Python implementation for "Multi-view co-similarity metric learning" paper

## Dependencies

- python>=3.7
- pytorch>=1.6.0
- torchvision>=0.8.1
- numpy>=1.19.2
- scikit-learn>=0.23.2
- scipy>=1.3.2
- python-igraph>=0.8.3
- leidenalg>=0.7.0
- h5py>=2.9.0

## Datasets

Four datasets used in the paper ([3-sources](http://mlg.ucd.ie/datasets/3sources.html), [BBCSport](http://mlg.ucd.ie/datasets/), [Caltech101-7](http://www.vision.caltech.edu/Image_Datasets/Caltech101/), [handwritten](https://archive.ics.uci.edu/ml/datasets/Multiple+Features), [ORL](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) and [Reuters](http://archive.ics.uci.edu/ml/datasets.html)) can be found in the data directory (*.mat format).

## Usage

### 1. Training MCML model

> train.py [-h] --dataset DATASET [--alpha ALPHA] [--beta BETA] [--tol_err TOL_ERR] [--maxIter MAXITER] [--random_state RANDOM_STATE] [--dev DEV] [--save_dir SAVE_DIR]

#### Arguments

> --dataset     training MCML model on which dataset, string, choose one from 3sources, BBCSport, Caltech7, handwritten, ORL or Reuters

#### Optional arguments

> -h, --help     show help message and exit
>
> --alpha     hyperparameter to weight each view, list, defaults to equally weighted
>
> --beta     hyperparameter for optimization, float, defaults to 1e-5
>
> --tol_err     tolerance error for checking convergence, float, defaults to 1e-5
>
> --maxIter     the maximum number of iterations, integer, defaults to 1000
>
> --random_state     random seed for reproducibility, integer, defaults to 0
>
> --dev     the device for training MCML, string, the default value is None ("cuda" if GPU is available else "cpu")
>
> --save_dir     the directory to save trained MCML model, string, defaults to "./results"

### 2. Clustering based on trained MCML model

> python clustering.py clustering.py [-h] --dataset DATASET --resolution RESOLUTION [--random_state RANDOM_STATE] [--model_dir MODEL_DIR]

#### Arguments

> --dataset     clustering on which dataset, string, choose one from 3sources, BBCSport, Caltech7, handwritten, ORL or Reuters
>
> --resolution     resolution for leiden algorithm, float, larger value results in more clusters

#### Optional arguments

> -h, --help     show help message and exit
>
> --random_state     random seed for reproducibility, integer, the default value is 0
>
> --model_dir     the directory for trained MCML model, string, the default value is "./results"

## Example usage

Example for training MCML model and clustering on BBCSport dataset with default optional arguments:

> python train.py --dataset BBCSport

> python clustering.py --dataset BBCSport --resolution 0.6

## Contributing

If you'd like to contribute, or have any suggestions for this GitHub repository, you can contact us at chuanchaozhang@163.com or open an issue here. 

All contributions welcome! All content in this repository is licensed under the BSD 3-Clause license.
