import uproot
from data.mnist import mnist_data_loader
from data.drellyan_gen import dygen_data_loader
from data.ttbar import ttbar_data_loader


def get_data( opts ): 

    if opts.data == "mnist":
        return mnist_data_loader()
    elif opts.data == 'dygen':
        return dygen_data_loader()
    elif opts.data == 'ttbar':
        return ttbar_data_loader()
