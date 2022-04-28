# MC simulations with WGAN 

Code to generate MC simulations with WGANs of t $\bar{t}$

## Set up the environment 

The `environment.yml` file contains the packages needed to run the code with pytorch and CUDA 10.2. 


```conda env create -f environment.yml```

```conda activate pytorch_v1_cuda_10_2```

## Examples with different datasets

### Generate images using the MNIST dataset

```python wgan.py --generator_iters 40000  --model convNNforNist --data mnist --trainingLabel mnisttraining  --do_what train --do_what generate```

### Generate Drell-Yan events using gen-level MC

```python wgan.py --generator_iters 100000  --model dense6inputs --data dygen --trainingLabel dytraining --do_what train --do_what generate --n_samples 10000 ```

### Train a GAN with ttbar events with bias

```nohup python wgan.py --generator_iters 80000 --model ttbarGAN_linear --data ttbar --trainingLabel ttbartraining --do_what train --constraint penalty --alpha 1e-05 --latent_space uniform --optimizer Adam --n_critic 8 --penalty_coeff 100 > ./test.txt &```

### Generate events from the trained GAN

```python wgan.py --model ttbarGAN_linear --data ttbar --do_what generate --save_samples pt --num_samples 37066 --num_model 42```

### Create plots comparing the generated sample with bias or original samples

```python wgan.py --data ttbar --num_model 42 --do_what plot --plot_opt bias_data --plot_opt density --plot_opt linear```


## Package contents

- `wgan.py`: main script that contains the training algorithm and the parsing of the different options.

- `models` directory: contains different architectures for the generator and critic networks, that is selected with the `--model` option. Associated to the critic is the dimensionality and distribution of the latent space, which is also defined here. 

- `data` directory: contains the scripts to handle data. It contains two example classes `drellyan_gen` and `mnist`, that are imported through `data_loaders`. In the context of this repository, data handling includes fetching the data, its preprocessing and its postprocessing, including production of plots. 


