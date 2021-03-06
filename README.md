# MC simulations with WGAN 

Code to train, generate and analyse MC simulations with WGANs of t t-bar process.

## Set up the environment 

The `environment.yml` file contains the packages needed to run the code with pytorch and CUDA 10.2. 


```conda env create -f environment.yml```

```conda activate pytorch_v1_cuda_10_2```

## Examples of use

### Train a GAN with ttbar events with bias
#### Execute the training (about 3h in cpu) in background and save the output to a test.txt file

``` nohup python wgan.py --model ttbarGAN_exp --do_what train --generator_iters 900000 --n_critic 5 --flip_iter 1 --batch_size 128 --optimizer RMSprop --alpha 5e-5 --alpha_end_factor 0.0001 --gen_coeff 2 --momentum 8e-8 --constraint clipping --clipping_value 0.02 --latent_space gaussian > ./lep_bjets_MET_B60_S20.txt &```

### Generate events from the trained GAN
#### Ignore output file

```nohup python wgan.py --model ttbarGAN_exp --latent_space gaussian --do_what generate --save_samples pt --n_samples 37066 --num_model 1 >/dev/null 2>&1 &```

### Create plots comparing the generated sample with bias or original samples

```python -W ignore wgan.py --do_what plot --plot_opt bias_data --plot_opt counts --plot_opt linear --plot_opt saveFig --num_model 1 > Stats.txt & tail -f Stats.txt```

```python -W ignore wgan.py --do_what plot --plot_opt BiasVSOriginal --plot_opt counts --plot_opt linear --plot_opt saveFig --num_model 1 > StatsBVSO.txt & tail -f StatsBVSO.txt```


## Package contents

- `wgan.py`: main script that contains the training algorithm and the parsing of the different options.

- `models` directory: contains different architectures for the generator and critic networks, that is selected with the `--model` option. Associated to the critic is the dimensionality and distribution of the latent space, which is also defined here. 

- `data` directory: contains the scripts to handle data. It contains two example classes `drellyan_gen` and `mnist`, that are imported through `data_loaders`. In the context of this repository, data handling includes fetching the data, its preprocessing and its postprocessing, including production of plots. 


