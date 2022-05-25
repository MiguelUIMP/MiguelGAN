# MC simulations with WGAN 

Code to train, generate and analyse MC simulations with WGANs of t t-bar process.

## Set up the environment 

The `environment.yml` file contains the packages needed to run the code with pytorch and CUDA 10.2. 


```conda env create -f environment.yml```

```conda activate pytorch_v1_cuda_10_2```

## Examples of use

### Train a GAN with ttbar events with bias
#### Execute the training (about 3h in cpu) in background and save the output to a test.txt file

```nohup python wgan.py --generator_iters 200000 --model ttbarGAN_linear --alpha 8e-5 --n_critic 6 --flip_iter 10000000 --batch_size 128 --optimizer RMSprop --alpha_end_factor 0.01 --gen_coeff 2 --momentum 1e-9 --do_what train --latent_space uniform --constraint clipping --clipping_value 0.025 > ./phi_eta_ptB20.txt &```

### Generate events from the trained GAN
#### Ignore output file

```nohup python wgan.py --model ttbarGAN_linear --do_what generate --save_samples pt --n_samples 37066 --num_model 107 >/dev/null 2>&1 &```

### Create plots comparing the generated sample with bias or original samples

```python wgan.py --do_what plot --plot_opt bias_data --plot_opt density --plot_opt linear --num_model 107```


## Package contents

- `wgan.py`: main script that contains the training algorithm and the parsing of the different options.

- `models` directory: contains different architectures for the generator and critic networks, that is selected with the `--model` option. Associated to the critic is the dimensionality and distribution of the latent space, which is also defined here. 

- `data` directory: contains the scripts to handle data. It contains two example classes `drellyan_gen` and `mnist`, that are imported through `data_loaders`. In the context of this repository, data handling includes fetching the data, its preprocessing and its postprocessing, including production of plots. 


