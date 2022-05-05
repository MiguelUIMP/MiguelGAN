# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:52:37 2022
@author: MiguelHoyo
"""

import torch
from torch import nn 

class DenseNNgenerator(torch.nn.Module):
    def __init__(self, channels, random_noise):
        super().__init__()
        self.main_module = nn.Sequential(
            
            nn.Linear((channels+random_noise), 300),
            #nn.BatchNorm1d(num_features=100),
            nn.ReLU(True),
            
            nn.Linear( 300, 150),
           # nn.BatchNorm1d(num_features=60),
            nn.ReLU(True),
            
            nn.Linear(150, 50),
            nn.ReLU(True),

            nn.Linear( 50, channels)
          
        )

    def forward(self, x):
        return self.main_module(x)

class DenseNNdiscriminator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            
            nn.Linear( channels, 60),
           # nn.BatchNorm1d(num_features=40),
            nn.ReLU(True),
            
            nn.Linear(60,40),
           # nn.BatchNorm1d(num_features=20),
            nn.ReLU(True),
            
            nn.Linear(40,1)
        )
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)




class ttbarGAN():


    def __init__(self, lat_space):
        channels = 18 # canales de salida de los datos, cuantas variables obtener
        random_noise = 100 # canales de ruido aleatorio que meter
        self.G=DenseNNgenerator(channels=channels, random_noise=random_noise)
        self.D=DenseNNdiscriminator(channels=channels)
        '''
        expresiones lambda en python:
            sirven para utilizar una funcion sin tener que definirla ni darla un nombre
            explicitamente, se llaman funciones anonimas tmb, uso
            
            lambda argumentos : expresion 
            
        '''    
        
        if lat_space == "gaussian":
            
            self.latent_space_generator = lambda batch_size : torch.randn(batch_size, random_noise)
        
        if lat_space == "uniform":
            
            self.latent_space_generator = lambda batch_size : torch.rand(batch_size, random_noise)
