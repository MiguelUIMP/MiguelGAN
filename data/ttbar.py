# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:03:48 2022

@author: MiguelHoyo
"""

import os, uproot, torch
import torch.utils.data as data
import uproot3_methods
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ttbar_TrainGen(data.Dataset):
    
    bias_folder = 'bias_ttbar'
    processed_bias_folder = 'processed_bias_ttbar'
    bias_file = 'bias_ttbar.pt'
    

    def __init__(self, root, download=False, transform=None):
        self.root = os.path.expanduser(root)
        self.transform=transform

        if download:
            self.download()

        '''
        automatizar el proceso de descarga de datos TODO
        '''
        
        if not self._check_exists():
            raise RuntimeError('Dataset with bias for train not found.')
        self.train_data = torch.load(
            os.path.join(self.root, self.processed_bias_folder, self.bias_file))

    def __getitem__(self, index):
        
        sample = self.train_data[index]
        if self.transform is not None:
            sample = self.transform(img)
        return sample

    def __len__(self):
        return len(self.train_data)


    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_bias_folder, self.bias_file)) 

    def download(self):
       # from six.moves import urllib
        import shutil

        transformed_file = "/home/ubuntu/addSystematics/ptLeptonB60S20/ttbar-Madgraph-MLM.root"
        if self._check_exists():
            return
        
        try:
            os.makedirs(os.path.join(self.root, self.bias_folder))
            os.makedirs(os.path.join(self.root, self.processed_bias_folder))
            shutil.copyfile(transformed_file, os.path.join(self.root, self.bias_folder, "bias_ttbar.root"))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        
        files=[os.path.join(self.root, self.bias_folder, "bias_ttbar.root")]
        
        # process and save as torch files
        print('Processing bias file...')
        training_set=read_root_files(files, fileType='train')

        with open(os.path.join(self.root, self.processed_bias_folder, self.bias_file), 'wb') as f:
            torch.save(training_set, f)
            



class ttbar_LatentGen(data.Dataset):

    bias_folder = 'bias_ttbar'
    unbias_folder = 'unbias_ttbar'
    processed_unbias_folder = 'processed_unbias_ttbar'
    unbias_file = 'unbias_ttbar.pt'
    

    def __init__(self, root, download=False, transform=None):
        self.root = os.path.expanduser(root)
        self.transform=transform

        if download:
            self.download()

        '''
        automatizar el proceso de descarga de datos TODO
        '''
        
        if not self._check_exists():
            raise RuntimeError('Dataset for latent space not found.')
        self.train_data = torch.load(
            os.path.join(self.root, self.processed_unbias_folder, self.unbias_file))

    def __getitem__(self, index):
        
        sample = self.train_data[index]
        if self.transform is not None:
            sample = self.transform(img)
        return sample

    def __len__(self):
        return len(self.train_data)


    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_unbias_folder, self.unbias_file))
    
    
    def latent_compare_path(self):
        return os.path.join(self.root, self.unbias_folder, "unbias_ttbar.root"), os.path.join(self.root, self.bias_folder, "bias_ttbar.root")
    

    def download(self):
       # from six.moves import urllib
        import shutil

        original_file = "/home/ubuntu/addSystematics/original/ttbar-Madgraph-MLM.root"
        if self._check_exists():
            return
        
        try:
            os.makedirs(os.path.join(self.root, self.unbias_folder))
            os.makedirs(os.path.join(self.root, self.processed_unbias_folder))
            shutil.copyfile(original_file, os.path.join(self.root, self.unbias_folder, "unbias_ttbar.root"))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
            
            
        files=[os.path.join(self.root, self.unbias_folder, "unbias_ttbar.root")]
        
        # process and save as torch files
        print('Processing unbias file...')
        training_set=read_root_files(files, fileType='latent')

        with open(os.path.join(self.root, self.processed_unbias_folder, self.unbias_file), 'wb') as f:
            torch.save(training_set, f)

        


def read_root_files(paths, fileType=None, generate=False, compare=False, process=True, pass_df=False):
    '''
    Read root files to return torch.tensor to latent space, train, generate samples, or pd.Dataframe to compare final results
    '''
    data=None
    # path is a list of paths to root files, all root files will be concat in <data> pd.DataFrame
    for path in paths:
        tf=uproot.open(path)
        tree=tf['t']
        if data is None: 
            data=tree.arrays(tree.keys(), library='pd')
        else:
            data=pd.concat([data,tree.arrays(tree.keys(), library='pd')])

        print("Dataframe shape: ", data.shape)
        # data es un pandas.DataFrame, data.values es un array, para una primera
        # aproximación, vamos a usar solo philep1, etalep1 y ptlep1 de las 34 posibles variables 
        # del dataframe
    
    '''
    # if variables need preprocess or not
    if process:
        processed_data = preProcess(data)
        # workaround due to in real life we dont have nu data neither mMET and etaMET
        if pass_df:
            return [var for var in processed_data.columns if var.find('nu') == -1 and var.find('mMET') == -1 and var.find('etaMET') == -1]
        # MC original dataset and bias dataset are split in 2 disjoint sets each one, it depends on their purpose
        if fileType=='train' or generate:
            # in the reshape (-1,3): -1 stands for the dataset size, keep it, 3 satands for the variables (columns) selected 
            return torch.reshape(torch.tensor(processed_data[:int(round(data.shape[0]/2))].values), (-1,18)) 

        if fileType=='latent':
            return torch.reshape(torch.tensor(processed_data[int(round(data.shape[0]/2)):].values), (-1,18))

        if compare:
            #return data[["philep1", "etalep1", "ptlep1"]][int(round(data.shape[0]/2)):]
            return data[[var for var in data.columns if var.find('nu') == -1 and var.find('mMET') == -1 and var.find('etaMET') == -1]][int(round(data.shape[0]/2)):]
    '''       

    # if variables need preprocess or not
    if process:
        processed_data = preProcess(data)
        # workaround due to in real life we dont have nu data neither mMET and etaMET
        if pass_df:
            return [var for var in processed_data.columns if var.find('nu') == -1 and var.find('m') == -1 and var.find('MET') == -1]
        # MC original dataset and bias dataset are split in 2 disjoint sets each one, it depends on their purpose
        if fileType=='train' or generate:
            # in the reshape (-1,3): -1 stands for the dataset size, keep it, 3 satands for the variables (columns) selected 
            return torch.reshape(torch.tensor(processed_data[:int(round(data.shape[0]/2))].values), (-1,12)) 

        if fileType=='latent':
            return torch.reshape(torch.tensor(processed_data[int(round(data.shape[0]/2)):].values), (-1,12))

        if compare:
            #return data[["philep1", "etalep1", "ptlep1"]][int(round(data.shape[0]/2)):]
            return data[[var for var in data.columns if var.find('nu') == -1 and var.find('MET') == -1 and var.find('m') == -1]][int(round(data.shape[0]/2)):]
 
        
    if not process:

        if fileType=='train' or generate:
            return torch.reshape(torch.tensor(data[var_to_use][:int(round(data.shape[0]/2))].values), (-1,18))

        if fileType=='latent':
            return torch.reshape(torch.tensor(data[var_to_use][int(round(data.shape[0]/2)):].values), (-1,18))

        if compare:
            return data[var_to_use][int(round(data.shape[0]/2)):]
            
def preProcess(data):
    '''
    Change from spherical transverse coordinates to cartesian 
    '''
    newData = None
    for var in ['lep1', 'lep2', 'b1', 'b2', 'MET']:
        # meter condicion para no usar la eta del MET, osea no sacar pzMET
        px = data[''.join(('pt', var))]*np.cos(data[''.join(('phi', var))])
        py = data[''.join(('pt', var))]*np.sin(data[''.join(('phi', var))])
        if var != 'MET':
            pz = data[''.join(('pt', var))]*np.sinh(data[''.join(('eta', var))]) 
        '''
        if newData is None:
            newData=pd.DataFrame({''.join(('px', var)): px, ''.join(('py', var)): py, ''.join(('pz', var)): pz, ''.join(('m', var)): data[''.join(('m', var))]})
            continue
        if newData is not None and var != 'MET':
            newData=pd.concat((newData, pd.DataFrame({''.join(('px', var)): px, ''.join(('py', var)): py, ''.join(('pz', var)): pz, ''.join(('m', var)): data[''.join(('m', var))]})), axis=1)
        if newData is not None and var == 'MET':
            newData=pd.concat((newData, pd.DataFrame({''.join(('px', var)): px, ''.join(('py', var)): py})), axis=1)
        '''
        if newData is None:
            newData=pd.DataFrame({''.join(('px', var)): px, ''.join(('py', var)): py, ''.join(('pz', var)): pz})
            continue
        if newData is not None and var != 'MET':
            newData=pd.concat((newData, pd.DataFrame({''.join(('px', var)): px, ''.join(('py', var)): py, ''.join(('pz', var)): pz})), axis=1)
        if newData is not None and var == 'MET':
            pass
        
    return newData
    


class ttbar_data_loader:

    def __init__(self):
        pass

    def get_data_loader(self, batch_size):
        training_set = ttbar_TrainGen( root='~/MiguelGAN', download=True, transform=None)
        assert training_set

        train_dataloader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

        return train_dataloader, None
    
    def get_latent_loader(self, batch_size):
        latent_set = ttbar_LatentGen( root='~/MiguelGAN', download=True, transform=None)
        assert latent_set

        train_dataloader = data.DataLoader(latent_set, batch_size=batch_size, shuffle=True)

        return train_dataloader, latent_set.latent_compare_path()
    
    

    def postProcess(self, samples, label):
        # in this case we will plot mll, but we could do something else
        mll=[]
        for samp in samples:
            samp=samp.numpy()
            lep1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim( samp[:,0], samp[:,1], samp[:,2], 0)
            lep2 = uproot3_methods.TLorentzVectorArray.from_ptetaphim( samp[:,3], samp[:,4], samp[:,5], 0)
            mll.append( (lep1+lep2).mass[0] ) 
        fig, ax = plt.subplots()
        ax.hist( mll , range =(0,500), bins=100) 
        plt.savefig('mll%s.png'%label)
        ax.clear()
        plt.close()

    def get_postProcessor(self):
        return lambda samples, label : self.postProcess(samples, label)
        

      



