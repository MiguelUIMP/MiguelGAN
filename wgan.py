import torch
from torch.autograd import Variable
import uproot
import os 
import numpy as np

from models.all_models import all_models
from data.data_loaders import get_data
from data.ttbar import read_root_files

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from numpy.random import randint
import pandas as pd

class WGAN_trainer:
    def __init__(self, opts):
        self._options=opts

        model_server               = all_models(self._options)
        self.G                     = model_server.G
        self.D                     = model_server.D
        self.generate_latent_space = model_server.latent_space_generator

        # these could be parsed in the options, but ok 
        self.n_critic=self._options.n_critic
        self.batch_size=self._options.batch_size
        self.optimizer=self._options.optimizer
        self.alpha=self._options.alpha
        self.momentum=self._options.momentum
        self.c=self._options.clipping_value
        self.generator_iters=self._options.generator_iters
        self.constraint = self._options.constraint
        self.penalty_coeff = self._options.penalty_coeff
        self.plot_opt = self._options.plot_opt


        # REAL DATA = DATA WITH BIAS
        # Get the real data previously saved
        # these are torch.utils.data.dataloader.DataLoader which will load Tensor dims batch_size x 34
        self.train_loader,_ = get_data(self._options).get_data_loader(self.batch_size)
        if opts.experiment == 'TTbar':
            self.latent_loader, (self.latent_path, self.compare_path) = get_data(self._options).get_latent_loader(self.batch_size)
            #aux_path = '/'.join(self.latent_path.split('/')[:-2])
            #self.compare_path = os.path.join(aux_path, "bias_ttbar", "bias_ttbar.root")
        '''
        
        Para generar muestras del evento TODO
        
        self.postProcessSamples=get_data(self._options).get_postProcessor()
        
        
        '''
        
        # Decide if we will (if we can) use the GPU
        self.cuda = self._options.cuda and torch.cuda.is_available()
        if not torch.cuda.is_available() and self._options.cuda:
            print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        if self.cuda:
            self.cuda_index=self._options.cuda_index
        
            
        if self.cuda: 
            self.G.cuda(self.cuda_index)
            self.D.cuda(self.cuda_index)


    def get_torch_variable(self, arg ):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)
        
    def trainIt(self):

        # Helper to create batches of date from a torch.utils.data.dataloader.DataLoader object
        def get_infinite_batches(data_loader):
            while True:
                for i, images in enumerate(data_loader):
                    print(i)
                    yield images
                    
        # generator objects, with .__next__() method returns a batch of loaded data 
        data = get_infinite_batches(self.train_loader)


        optim_discriminator = torch.optim.RMSprop( self.D.parameters(), lr=self.alpha)
        optim_generator     = torch.optim.RMSprop( self.G.parameters(), lr=self.alpha)

        values_g_loss_data     =[]
        values_d_loss_fake_data=[]
        values_d_loss_real_data=[]
        for g_iter in range(self.generator_iters):
                
            for p in self.D.parameters():
                p.requires_grad=True
            for t in range(self.n_critic):
                self.D.zero_grad() # we need to set the gradients to zero, otherwise pytorch will sum them every time we call backward
                images=data.__next__()
                if (images.size()[0] != self.batch_size): # the dataset may not be multiple of the batch size
                    continue
                real_data=self.get_torch_variable(images)
                fake_data=self.get_torch_variable( self.generate_latent_space(self.batch_size )) # TODO: the latent space is hardcoded, should be an input (use a lambda function in the models.)
                loss_a=torch.mean(self.D(real_data)-self.D(self.G(fake_data)))
                loss_a.backward() # compute gradients 
                optim_discriminator.step() # move the parameters

                # clip the parameters
                for p in self.D.parameters():
                    p.data.clamp_(-self.c,self.c)


                # Get the components of the loss to store them 
                loss_a_real_data=torch.mean(self.D(real_data)).data.cpu()
                loss_a_fake_data=torch.mean(-self.D(self.G(fake_data))).data.cpu()

                print(f'  Discriminator iteration: {t}/{self.n_critic}, loss_fake: {loss_a_fake_data}, loss_real: {loss_a_real_data}')
            

            # to avoid computation
            for p in self.D.parameters():
                p.requires_grad = False  
            
            self.G.zero_grad()  # we need to set the gradients to zero, otherwise pytorch will sum them every time we call backward

            fake_data=self.get_torch_variable(self.generate_latent_space(self.batch_size))
            loss_b=torch.mean(self.D(self.G(fake_data))) # because the gradient then goes with a minus
            loss_b.backward()
            optim_generator.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {loss_b.data.cpu()}')

            # to plot 
            values_g_loss_data     .append(loss_b.data.cpu() )
            values_d_loss_fake_data.append(loss_a_fake_data   )
            values_d_loss_real_data.append(loss_a_real_data   )

            if not (g_iter%1000):
                self.save_model(label=f"gen_iter_{g_iter}")
                
                '''
                
                Para generar eventos TODO
                
                self.generate_samples(self._options.n_samples, label=f"gen_iter_{g_iter}", load_model=False)

                '''
        
        fig, ax = plt.subplots()
        plot1=ax.plot( range(len(values_g_loss_data     )),values_g_loss_data      , label='loss generator')
        plot2=ax.plot( range(len(values_d_loss_fake_data)),values_d_loss_fake_data , label='loss critic fake data')
        plot3=ax.plot( range(len(values_d_loss_real_data)),values_d_loss_real_data , label='loss critic real data')
        plt.legend(handles=[plot1[0],plot2[0],plot3[0]])
        plt.savefig('training_%s.png'%self._options.trainingLabel)
        ax.clear()
        plt.close()
        self.save_model()
        
        
    def trainTTbar(self):

        # Helper to create batches of date from a torch.utils.data.dataloader.DataLoader object
        def get_infinite_batches(data_loader):
            while True:
                for i, images in enumerate(data_loader):
                    #print("mini-batch:",i)
                    yield images
                    
        # generator objects, with .__next__() method returns a batch of loaded data 
        data = get_infinite_batches(self.train_loader)
        latent = get_infinite_batches(self.latent_loader)

        if self.optimizer == 'RMSprop':
            optim_discriminator = torch.optim.RMSprop( self.D.parameters(), lr=self.alpha, momentum=self.momentum)
            optim_generator     = torch.optim.RMSprop( self.G.parameters(), lr=self.alpha, momentum=self.momentum)
            
        if self.optimizer == 'Adam':
            # Default values for (beta_1, beta2_)=(0.9, 0.999), paper WGAN-GP values (beta_1, beta_2)=(0, 0.9)
            optim_discriminator = torch.optim.Adam( self.D.parameters(), lr=self.alpha, betas=(0, 0.9))  
            optim_generator     = torch.optim.Adam( self.G.parameters(), lr=self.alpha, betas=(0, 0.9))
            
        if self.optimizer == 'SGD':
            optim_discriminator = torch.optim.SGD( self.D.parameters(), lr=self.alpha, momentum=self.momentum)
            optim_generator     = torch.optim.SGD( self.G.parameters(), lr=self.alpha, momentum=self.momentum)

        values_g_loss_data     =[]
        values_d_loss_fake_data=[]
        values_d_loss_real_data=[]
        
        def early_stopping(d_r, d_f, g, g_it, bound=0.002):
            val_d_r = np.asarray([d_r[ind] for ind in [g_it-4000, g_it-3000, g_it-2000, g_it-1000, g_it]], dtype="float32")
            val_d_f = np.asarray([d_f[ind] for ind in [g_it-4000, g_it-3000, g_it-2000, g_it-1000, g_it]], dtype="float32")
            val_g = np.asarray([g[ind] for ind in [g_it-4000, g_it-3000, g_it-2000, g_it-1000, g_it]], dtype="float32")
            upper_d_r = np.mean(val_d_r) + bound
            lower_d_r = np.mean(val_d_r) - bound
            upper_d_f = np.mean(val_d_f) + bound
            lower_d_f = np.mean(val_d_f) - bound
            upper_g = np.mean(val_g) + bound
            lower_g = np.mean(val_g) - bound
            return ( any(val_d_r>upper_d_r) or any(val_d_r<lower_d_r) or any(val_d_f>upper_d_f) or any(val_d_f<lower_d_f) or any(val_g>upper_g) or any(val_g<lower_g) )

        
        for g_iter in range(self.generator_iters):
                
            for p in self.D.parameters():
                p.requires_grad=True
            for t in range(self.n_critic):
                self.D.zero_grad() # we need to set the gradients to zero, otherwise pytorch will sum them every time we call backward
                images=data.__next__()
                images_lat=latent.__next__()
                if (images.size()[0] != self.batch_size or images_lat.size()[0] != self.batch_size): # the dataset may not be multiple of the batch size
                    continue
                real_data=self.get_torch_variable(images)
                fake_data=self.get_torch_variable( torch.cat( (self.generate_latent_space(self.batch_size), images_lat), dim=1 ) ) # TODO: the latent space is hardcoded, should be an input (use a lambda function in the models.)

                if self.constraint == "clipping":
                    loss_a= -torch.mean(self.D(real_data)) + torch.mean(self.D(self.G(fake_data)))
                    loss_a.backward() # compute gradients 
                    optim_discriminator.step() # move the parameters
                    
                    # clip the parameters
                    for p in self.D.parameters():
                        p.data.clamp_(-self.c,self.c)
                        
                if self.constraint == "penalty":
                    grad_penalty = self.gradient_penalty(real_imgs=real_data, fake_imgs=self.G(fake_data), penalty_coeff=self.penalty_coeff)
                    loss_a= -torch.mean(self.D(real_data)) + torch.mean(self.D(self.G(fake_data))) + grad_penalty
                    loss_a.backward() # compute gradients 
                    optim_discriminator.step() # move the parameters

                # Get the components of the loss to store them 
                loss_a_real_data=-torch.mean(self.D(real_data)).data.cpu()
                loss_a_fake_data=torch.mean(self.D(self.G(fake_data))).data.cpu()

                print(f'  Discriminator iteration: {t}/{self.n_critic}, loss_fake: {loss_a_fake_data}, loss_real: {loss_a_real_data}')
            

            # to avoid computation
            for p in self.D.parameters():
                p.requires_grad = False  
            
            self.G.zero_grad()  # we need to set the gradients to zero, otherwise pytorch will sum them every time we call backward
            images_lat=latent.__next__()
            while (images_lat.size()[0] != self.batch_size):
                images_lat=latent.__next__()
            fake_data=self.get_torch_variable( torch.cat( (self.generate_latent_space(self.batch_size), images_lat), dim=1 ) ) # TODO: the latent space is hardcoded, should be an input (use a lambda function in the models.)
            loss_b=-torch.mean(self.D(self.G(fake_data))) # because the gradient then goes with a minus
            loss_b.backward()
            optim_generator.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {loss_b.data.cpu()}')

            # to plot 
            values_g_loss_data     .append(loss_b.data.cpu() )
            values_d_loss_fake_data.append(loss_a_fake_data   )
            values_d_loss_real_data.append(loss_a_real_data   )
            
            #if not (g_iter%1000):
            #    self.save_model(label=f"gen_iter_{g_iter}")
            
            # early stopping condition check, to avoid waste of time if the GAN loss functions get stuck without improvements
            if not (g_iter%5000) and g_iter!=0:
                if not early_stopping(values_d_loss_real_data, values_d_loss_fake_data, values_g_loss_data, g_iter):
                    print("GAN loss function got stuck")
                    break
                    
                

        model_lab = self.save_model(label="FINAL")

        fig, ax = plt.subplots()
        plot3=ax.plot( range(len(values_d_loss_real_data)),values_d_loss_real_data , label='loss critic real data')
        plot2=ax.plot( range(len(values_d_loss_fake_data)),values_d_loss_fake_data , label='loss critic fake data')
        plot1=ax.plot( range(len(values_g_loss_data     )),values_g_loss_data      , label='loss generator')
        plt.legend(handles=[plot3[0],plot2[0],plot1[0]])
        plt.savefig(f'./TrainedGANs/LossFunction_{model_lab}.png')
        ax.clear()
        plt.close()
        

    def gradient_penalty(self, real_imgs, fake_imgs, penalty_coeff=10):
        #self.G.eval()
        #self.D.eval()
        epsilon = torch.rand(self.batch_size, 1)
        epsilon = epsilon.expand_as(real_imgs)
        
        interpolation = epsilon * real_imgs.data + (1 - epsilon) * fake_imgs.data
        interpolation = Variable(interpolation, requires_grad=True)
        

        interpolation_logits = self.D(interpolation)
        grad_outputs = torch.ones(interpolation_logits.size())

        gradients = torch.autograd.grad(outputs=interpolation_logits,
                                inputs=interpolation,
                                grad_outputs=grad_outputs,
                                create_graph=True,
                                retain_graph=True)[0]

        gradients = gradients.view(self.batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        grad_penal =  torch.mean(penalty_coeff * ((gradients_norm - 1) ** 2))
        #self.G.train()
        #self.D.train()
        return grad_penal

    
    def save_model(self,label=""):
        
        if os.path.exists("./TrainedGANs"):
            pass
        else:
            try:
                os.makedirs("./TrainedGANs")
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise
        num_docs = len(os.listdir("./TrainedGANs"))
        label = '_'.join((label, str(int(num_docs/3))))
        torch.save(self.G.state_dict(), f'./TrainedGANs/{self._options.trainingLabel}_generator_{label}.pkl')
        torch.save(self.D.state_dict(), f'./TrainedGANs/{self._options.trainingLabel}_discriminator_{label}.pkl')
        with open("./TrainedGANs.txt", 'a+') as f:
            f.write(''.join(('\n', label, ': ', str(vars(self._options)))))
        print(f'Models save in ./TrainedGANs to {self._options.trainingLabel}_discriminator_{label}.pkl & {self._options.trainingLabel}_generator_{label}.pkl')
        return label

    def load_model(self,label=""):
        # usually postprocessing is done in the cpu, but could be customized in the future
        self.G.load_state_dict(torch.load(f'./TrainedGANs/{self._options.trainingLabel}_generator_{label}.pkl',map_location=torch.device('cpu')))
        self.D.load_state_dict(torch.load(f'./TrainedGANs/{self._options.trainingLabel}_discriminator_{label}.pkl',map_location=torch.device('cpu'))) 

    
    def generate_samples(self, number_of_samples, save_as, num_model, label="FINAL" ):
        if num_model is None:
            num_docs = len(os.listdir("./TrainedGANs"))
            label = '_'.join((label, str(int(num_docs/3)-1)))
        else :
            label = '_'.join((label, str(num_model)))
        
        self.load_model(label)
        latent_data = read_root_files([self.latent_path], generate=True) 
        samples=[]
        for _ in range(number_of_samples):
            z=self.get_torch_variable( torch.cat( (self.generate_latent_space(1), torch.reshape(latent_data[randint(low=0, high=latent_data.shape[0]), :], (-1,3))), dim=1 ) )
            self.G.eval()
            sample=self.G(z).data.cpu()
            samples.append( sample ) 
            print(sample)
        
        
        samples_tensor = torch.squeeze(torch.stack(samples))
        
        self.save_samples(samples_tensor, label, toPytorch=(save_as=='pt' or save_as=='all'), toRoot=(save_as=='root' or save_as=='all'))
        
    
    def save_samples(self, samples_tensor, label, toPytorch=True, toRoot=True):

        path = '/'.join(self.latent_path.split('/')[:-2])
        
        if os.path.exists(os.path.join(path, "./GeneratedSamplesTTbar")):
            pass
        else:
            try:
                os.makedirs(os.path.join(path, "./GeneratedSamplesTTbar"))
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise
        
        if toPytorch:
            with open(os.path.join(path, f'./GeneratedSamplesTTbar/samples_{label}.pt'), 'wb') as f:
                torch.save(samples_tensor, f)
            print("Samples successfully saved in:", os.path.join(path, f'./GeneratedSamplesTTbar/samples_{label}.pt'))
        
        if toRoot:
            with uproot.recreate(os.path.join(path, f'./GeneratedSamplesTTbar/samples_{label}.root')) as f:
                df = pd.DataFrame(samples_tensor).astype("float")
                df.columns=['philep1', 'etalep1', 'ptlep1']
                f['t'] = df
            print("Samples successfully saved in:", os.path.join(path, f'./GeneratedSamplesTTbar/samples_{label}.root'))
                
                
                
    def plot_samples(self, plot_options, num_model, label="FINAL", process=False):
        
        # load last model if not model number is passed
        if num_model is None:
            num_docs = len(os.listdir("./TrainedGANs"))
            label = '_'.join((label, str(int(num_docs/3)-1)))
        else :
            label = '_'.join((label, str(num_model)))
            
        if not os.path.exists(f'./GeneratedSamplesTTbar/samples_{label}.pt') and num_model is not None:
            raise RuntimeError('Introduce a valid samples model number, current value does not exist.')
        if not os.path.exists(f'./GeneratedSamplesTTbar/samples_{label}.pt') and num_model is None:
            raise RuntimeError('Last model saved has not samples generated, generate samples before to plot them.')
            
        # samples tensor load to cpu, it could be load in the gpu    
        samples_tensor = torch.load(f'./GeneratedSamplesTTbar/samples_{label}.pt', map_location=torch.device('cpu'))
        
        # if samples need postProcess to change from cartesian to spherical basis or are already load in spherical ones
        if process:
            samples_df = self.postProcess(samples_tensor)
        if not process:
            samples_df = pd.DataFrame(data=samples_tensor.numpy(), columns=["philep1", "etalep1", "ptlep1"], dtype="float64")
            
        samples_mean = round(samples_df.mean(),2)
        samples_std = round(samples_df.std(),2)
        
        data_type=[]
        plot_type=[]
        scale_type=[]
        
        if "original_data" in plot_options:
            data_type.append("original data")
        if "bias_data" in plot_options:
            data_type.append("biased data")
        if "density" in plot_options:
            plot_type.append(True)
        if "counts" in plot_options:
            plot_type.append(False)
        if "log" in plot_options:
            scale_type.append("log")
        if "linear" in plot_options:
            scale_type.append("linear")
            
        for data_t in data_type:
            if data_t == "original_data":
                compare_df = read_root_files([self.latent_path], compare=True)
            if data_t == "bias_data":
                compare_df = read_root_files([self.compare_path], compare=True)
                
            compare_mean = round(compare_df.mean(),2)
            compare_std = round(compare_df.std(),2)

            for plot_t in plot_type:
                for scale_t in scale_type:
         
                    for var in samples_df:

                        plt.figure(figsize=(11,8));
                        hist_range_com = (compare_df.min()[var], compare_df.max()[var])
                        plt.hist(compare_df[var] , range=hist_range_com, bins=200, density=plot_t, alpha=0.5, label=f'MC simulation {data_t}; mean: {compare_mean[var]} std: {compare_std[var]}');
                        hist_range_sam = (samples_df.min()[var], samples_df.max()[var])
                        plt.hist(samples_df[var], range=hist_range_sam, bins=200, density=plot_t, alpha=0.5, label=f'Generated samples; mean: {samples_mean[var]} std: {samples_std[var]}');
                        if hist_range_sam[0]<hist_range_com[0]
                            font = {'family': 'serif',
                                    'color':  'darkred',
                                    'weight': 'normal',
                                    'size': 12,
                                    }
                            out_b=round(hist_range_sam[0], 2)
                            plt.text(0.55, 0.75, "\n".join((f'', f'sample out of lower bound up to {out_b}')), fontdict=font, transform=plt.gca().transAxes)
                        if hist_range_sam[1]>hist_range_com[1]
                            font = {'family': 'serif',
                                    'color':  'darkred',
                                    'weight': 'normal',
                                    'size': 16,
                                    }
                            out_b=round(hist_range_sam[1], 2)
                            plt.text(0.55, 0.75, f'sample out of upper bound up to {out_b}', fontdict=font, transform=plt.gca().transAxes)
                        plt.xlabel("ptlep1")
                        
                        if plot_t:
                            plt.ylabel("Density")
                        if not plot_t:
                            plt.ylabel("Counts")
                        plt.yscale(scale_t)
                        plt.title(f'{var} for generated and simulated MC samples')
                        plt.legend(loc='upper right')
                        plt.savefig(f'./GeneratedSamplesTTbar/comparation_{label}_{var}.png')
                        plt.close()
        
    def postProcess(self, samples_tensor):
        
        samples = pd.DataFrame(data=samples_tensor.numpy(), columns=["pxlep1", "pylep1", "pzlep1"], dtype="float64")
        phi = np.arctan(samples['pylep1']/samples['pxlep1'])
        pt = np.sqrt(samples['pylep1']**2+samples['pxlep1']**2)
        eta = np.arcsinh(samples['pzlep1']/pt)
        
        return pd.DataFrame({'philep1': phi, 'etalep1': eta, 'ptlep1': pt})
        


if __name__=="__main__":

    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("-m", "--model",           dest="model", type="string", default="convNNforNist", help="Architecture of the generator and critic. It also fixes the latent space distribution");
    parser.add_option("--experiment",           dest="experiment", type="string", default="TTbar", help="experiment to load models");
    parser.add_option("--n_critic",           dest="n_critic", type="int", default=5, help="Number of iterations of the critic per generator iteration");
    parser.add_option("--generator_iters",           dest="generator_iters", type="int", default=40000, help="Number of generator iterations");
    parser.add_option("--batch_size",           dest="batch_size", type="int", default=64, help="Mini-batch size");
    parser.add_option("--optimizer",           dest="optimizer", type="string", default="RMSprop", help="Optimizer to use, RMSprop, Adam or SGD");
    parser.add_option("--alpha",           dest="alpha", type="float", default=0.00005, help="Learning rate");
    parser.add_option("--momentum",           dest="momentum", type="float", default=0, help="Momentum");
    parser.add_option("--clipping_value",           dest="clipping_value", type="float", default=0.01, help="Clipping parameters of the discriminator between (-c,c)");
    parser.add_option("--data",           dest="data", type="string", default='mnist', help="Dataset to train with");
    parser.add_option("--no-cuda",           dest="cuda", action='store_false', default=True, help="Do not try to use cuda. Otherwise it will try to use cuda only if its available");
    parser.add_option("--cuda_index",           dest="cuda_index", type="int", default=0, help="Index of the device to use");
    parser.add_option("--do_what",           dest="do_what", action='append', type="string", default=[], help="What to do");
    parser.add_option("--trainingLabel",           dest="trainingLabel",  type="string", default='trainingv1', help="Label where store to/read from the models");
    parser.add_option("--n_samples",           dest="n_samples",  type="int", default=12, help="Number of samples to be generated");
    parser.add_option("--save_samples",           dest="save_samples",  type="string", default="all", help="How to save the samples generated: in .pt (use pt), in .root (use root), both (use all), or do not save (use none)");
    parser.add_option("--latent_space",           dest="latent_space",  type="string", default="gaussian", help="use uniform to sample from uniform dist in latent space or gaussian to follow suit from gaussian dist");
    parser.add_option("--num_model",           dest="num_model",  type="int", default=None, help="model number from which load the data to generate sampels or plot samples");
    parser.add_option("--constraint",           dest="constraint",  type="string", default="clipping", help="Lipschitz constraint, use clipping weights or gradient penalty");
    parser.add_option("--penalty_coeff",           dest="penalty_coeff",  type="float", default=10.0, help="Gradient penalty coefficient");
    parser.add_option("--plot_opt",           dest="plot_opt", action='append', type="string", default=[], help="plot histogram with log_scale or normal_scale, counts or density, bias_data or MC_data");


    (options, args) = parser.parse_args()

    model = WGAN_trainer(options)
    if ('train' in options.do_what) & (('TTbar') not in options.experiment):
        model.trainIt()
        
    if ('train' in options.do_what) & (('TTbar') in options.experiment):
        model.trainTTbar()
        
    if 'generate' in options.do_what:
        model.generate_samples(options.n_samples, options.save_samples, options.num_model)
      
    if 'plot' in options.do_what:
        model.plot_samples(options.plot_opt, options.num_model)
        
    
        
###################
#
#   Typical use of the ttbar code:
#   
#   python wgan.py --generator_iters 100 --model ttbarGAN --data ttbar --trainingLabel TTbar --do_what train
#
###################
