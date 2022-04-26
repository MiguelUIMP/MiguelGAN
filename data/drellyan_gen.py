import os, uproot, torch
import torch.utils.data as data
import uproot3_methods
import matplotlib.pyplot as plt
import pandas as pd
class DrellYan_GenLevel(data.Dataset):
    urls = [
        ("https://cernbox.cern.ch/index.php/s/k7PCvm3U4fpHvhO/download","DYJets_1.root"),
        ("https://cernbox.cern.ch/index.php/s/mMPCESPDWyqO8ra/download","DYJets_2.root"),
        ("https://cernbox.cern.ch/index.php/s/qMnl4JfZJkwaTBE/download","DYJets_3.root"),
    ]
    raw_folder = 'raw_dy'
    processed_folder = 'processed_dy'
    training_file = 'training_dy.pt'

    def __init__(self, root, download=False, transform=None):
        self.root = os.path.expanduser(root)
        self.transform=transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self.train_data = torch.load(
            os.path.join(self.root, self.processed_folder, self.training_file))

    def __getitem__(self, index):
        
        sample = self.train_data[index]
        if self.transform is not None:
            sample = self.transform(img)
        return sample

    def __len__(self):
        return len(self.train_data)


    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) 

    def download(self):
        from six.moves import urllib

        if self._check_exists():
            return
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        files=[]
        for url in self.urls:
            print('Downloading ' + url[0])
            data = urllib.request.urlopen(url[0])
            file_path = os.path.join(self.root, self.raw_folder, url[1])
            with open(file_path, 'wb') as f:
                f.write(data.read())
            files.append(file_path)


        # process and save as torch files
        print('Processing...')
        training_set=read_root_files(files)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)


def read_root_files(paths):
    data=None
    for path in paths:
        tf=uproot.open(path)
        tree=tf['Events']
        if data is None: 
            data=tree.arrays(tree.keys(), library='pd')
        else:
            data=pd.concat([data,tree.arrays(tree.keys(), library='pd')])

        print(data.size)
    return torch.tensor(data.values)


class dygen_data_loader:

    def __init__(self):
        pass

    def get_data_loader(self, batch_size):
        training_set = DrellYan_GenLevel( root='', download=True, transform=None)
        assert training_set

        train_dataloader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

        return train_dataloader, None

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
        

    

