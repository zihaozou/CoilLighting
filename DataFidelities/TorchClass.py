'''
Class for quadratic-norm on subsampled 2D Fourier measurements
Mingyang Xie, CIG, WUSTL, 2020
Based on MATLAB code by U. S. Kamilov, CIG, WUSTL, 2017
'''

import numpy as np
from tqdm import tqdm
import random
import time
from DataFidelities.pytorch_radon.filters import HannFilter
from DataFidelities.pytorch_radon import Radon, IRadon
import torch
from utils.util import *
import scipy.io as sio

###################################################
###                   Tomography Class                  ###
###################################################

class TorchClass(nn.Module):

    def __init__(self, sigSize):
        super(TorchClass, self).__init__()              

        self.sigSize = sigSize
        self.device = 'cuda'

    def size(self):
        sigSize = self.sigSize
        return sigSize

    def res(self, x, theta):
        z = self.fmult(x, theta)
        return z-self.y

    def grad_theta_full(self, x, y, theta):
        
        theta = theta.requires_grad_(True)
        # print('***************************')
        # print(x.shape, y.shape, theta.shape)
        # print('***************************')
        r = Radon(self.sigSize, theta, False, dtype=torch.float, device=self.device).to(self.device)
        sino_batch = r(x)
        # print(sino_batch.shape)
        # print('***************************')
        # loss
        loss_torch = torch.nn.MSELoss(reduction='sum')(sino_batch, y).to(self.device)
        # print(loss_torch)
        # print('***************************')

        loss_torch.backward(retain_graph=True)
        theta_grad = theta.grad
        return theta_grad

    def grad(self, x, y, theta):
        
        with torch.no_grad():
            gradList = []
            for i in range(x.shape[0]):
                gradList.append(self.ftran(self.fmult(x[i,:].unsqueeze(0), theta[i].squeeze()) - y[i,:].unsqueeze(0), theta[i].squeeze()))
            g = torch.cat(gradList, 0)    

        return g

    def gradStoc(self, x, theta):
        pass

    def fmult(self, x, theta):

        device = x.device
        r = Radon(self.sigSize, theta, False, dtype=torch.float, device=device)
        sino = r(x)

        return sino
    
    def ftran(self, z, theta):

        device = z.device
        ir = IRadon(self.sigSize, theta, False, dtype=torch.float, use_filter=None, device=device)
        reco_torch = ir(z)

        return reco_torch

    @staticmethod
    def tomoCT(ipt, sigSize, device=None, batchAngles=None, numAngles=180, inputSNR=40):

        if batchAngles is not None:
            print('In the tomoCT function of TorchClass, the batchAngles parameteris currently unsupported.')
            exit()
        device = ipt.device
        # generate angle array
        theta = np.linspace(0., 180, numAngles, endpoint=False)
               
        # convert to torch
        theta = torch.tensor(theta, dtype=torch.float, device=device)
        ipt = ipt.unsqueeze(0).unsqueeze(0).to(device)
        
        # forward project
        r = Radon(sigSize, theta, False, device=device).to(device)
        sino = r(ipt.to(device))

        # add white noise to the sinogram
        sino = addwgn_torch(sino, inputSNR)

        # backward project
        ir = IRadon(sigSize, theta, False, use_filter=None, device=device).to(device)
        recon_bp = ir(sino)
        # filtered backward project
        ir_hann = IRadon(sigSize, theta, False, dtype=torch.float, use_filter=HannFilter(), device=device).to(device)
        reco_fbp_hann = ir_hann(sino)
        reco_fbp_hann[reco_fbp_hann<=0] = 0
        
        return sino, recon_bp, reco_fbp_hann, theta






