import torch
from models.convgann import ConvGAN
from models.dense_6inputs import Dense6inputs
from models.ttbar_gan import ttbarGAN as ttbarGAN_original
from models.ttbar_gan_expanded import ttbarGAN as ttbarGAN_exp 
from models.ttbar_gan_conv import ttbarGAN as ttbarGAN_conv 
from models.ttbar_gan_linear import ttbarGAN as ttbarGAN_linear 



def all_models(opts):
    
    if opts.model == 'convNNforNist':
        return ConvGAN()
    if opts.model == "dense6inputs":
        return Dense6inputs()
    if opts.model == "ttbarGAN_original":
        return ttbarGAN_original(opts.latent_space)
    if opts.model == "ttbarGAN_exp":
        return ttbarGAN_exp(opts.latent_space)
    if opts.model == "ttbarGAN_conv":
        return ttbarGAN_conv(opts.latent_space)
    if opts.model == "ttbarGAN_linear":
        return ttbarGAN_linear(opts.latent_space)

