import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.autograd as autograd
import torch.functional
from math import pi,log
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import io
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
import torch.optim as optim
import json
from DataFidelities.pytorch_radon.radon import Radon,IRadon
from DataFidelities.pytorch_radon.filters import HannFilter
from math import sqrt
from skimage.metrics import peak_signal_noise_ratio
from random import gauss
class LinearBlock(nn.Module):
    def __init__(self,numInput,numOutput,batchNorm,acti):
        super(LinearBlock,self).__init__()
        actiDict={
                'relu':nn.LeakyReLU,
                'sigmoid':nn.Sigmoid,
                'tanh':nn.Tanh
                }
        modules=[nn.Linear(numInput,numOutput)]
        if batchNorm==True:
            modules.append(nn.BatchNorm1d(numOutput))
        modules.append(actiDict[acti]())
        self.block=nn.Sequential(*modules)
        self.block.apply(self.init_weights)
    def forward(self,x):
        return self.block(x)
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight,a=0.01)
            m.bias.data.fill_(1)

class FFM(nn.Module):
    def __init__(self,mode,L,numInput):
        super(FFM,self).__init__()
        self.mode=mode
        self.L=L
        stepTensor=torch.arange(end=L,start=0).repeat_interleave(2*numInput)
        index=torch.arange(start=0,end=L*numInput*2,step=numInput*2)
        ffm=torch.empty(L*2*numInput)
        if mode=='linear':
            stepTensor+=1
            for i in range(numInput):
                ffm[index+i]=.5*stepTensor[index]*pi
                ffm[index+numInput+i]=.5*stepTensor[index]*pi
        elif mode=='loglinear':
            for i in range(numInput):
                ffm[index+i]=2**stepTensor[index]*pi
                ffm[index+numInput+i]=2**stepTensor[index]*pi
        ffm=ffm.unsqueeze(0)
        self.register_buffer('ffmModule',ffm)
    def forward(self,x):
        repeatX=x.repeat(1,self.L*2)
        repeatX=repeatX*self.ffmModule
        repeatX[:,::4]=torch.sin(repeatX[:,::4])
        repeatX[:,1::4]=torch.sin(repeatX[:,1::4])
        repeatX[:,2::4]=torch.cos(repeatX[:,2::4])
        repeatX[:,3::4]=torch.cos(repeatX[:,3::4])
        return repeatX

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz = x0.shape[0]
    numEle=int(torch.numel(x0)/bsz)
    X = torch.zeros(bsz, m, numEle, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, numEle, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res


class DEQ(nn.Module):
    def __init__(self,
                addx,
                numskipInput,
                numInput=40,
                numEncodeNeurons=256,
                numExpandLayers=5,
                skipStride=2,
                norm=False,
                acti='relu',
                lastActi='sigmoid'):
        super(DEQ,self).__init__()
        currNumInput=numInput
        currNumOutput=numEncodeNeurons
        if skipStride>=numExpandLayers:
            raise ValueError('skipStride should be smaller than numExpandlayers')
        self.skipList=list(range(0,numExpandLayers,skipStride+1))
        expandLst=[]
        for i in range(numExpandLayers):
            if i+1==addx or i==numExpandLayers-1:
                currNumOutput=numInput
            if i in self.skipList:
                expandLst.append(LinearBlock(currNumInput+numskipInput,currNumOutput,norm,acti))
            elif i==numExpandLayers-1:
                expandLst.append(LinearBlock(currNumInput,currNumOutput,norm,lastActi))
            else:
                expandLst.append(LinearBlock(currNumInput,currNumOutput,norm,acti))
            currNumInput=currNumOutput
            currNumOutput=numEncodeNeurons
        self.expandLayers=nn.ModuleList(expandLst)
        self.skipStride=skipStride
        self.addx=addx
    def forward(self,z,x,skip,deq):
        if deq:
            curr=z
            for i,l in enumerate(self.expandLayers):
                if i==self.addx:
                    curr=curr+x
                if i in self.skipList:
                    curr=l(torch.cat((curr,skip),dim=1))
                else:
                    curr=l(curr)
        else:
            curr=x
            for i,l in enumerate(self.expandLayers):
                if i in self.skipList:
                    curr=l(torch.cat((curr,skip),dim=1))
                else:
                    curr=l(curr)
        return curr

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver,**kwargs):
        super(DEQFixedPoint,self).__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        self.lastz=None
    def forward(self, x,skip,deq):
        # compute forward pass and re-engage autograd tape
        if deq:
            with torch.no_grad():
                z, self.forward_res = self.solver(lambda z : self.f(z, x,skip,deq), self.lastz if self.lastz is not None else torch.zeros_like(x), **self.kwargs)
            newz = self.f(z.requires_grad_(),x,skip,deq)
            self.lastz=newz.clone().detach()
            if self.training:
                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()
                    g, self.backward_res = self.solver(lambda y : autograd.grad(newz, z, y, retain_graph=True)[0] + grad,
                                                    grad, **self.kwargs)
                    return g
                self.hook=newz.register_hook(backward_hook)
        else:
            self.lastz=None
            newz=self.f(None, x,skip,deq)
        return newz
    




class CoilwithDEQBase(pl.LightningModule):
    def __init__(self,
                lr=.1,
                batch_size=65250,
                L=10,
                numInput=2,
                XPath='data/X.csv',
                yPath='data/y.csv',
                originImagePath='data/image.csv',
                fullXPath='data/fullX.csv',
                fullyPath='data/fully.csv',
                originFullyPath='data/originFullyy.csv',
                normalized=False,
                tmax=1000,
                **kwargs
                ):
        super().__init__()
        self.save_hyperparameters()
        self.ffm=FFM('loglinear',L,numInput)
        self.lossFunc=nn.MSELoss()
        self.DEQFixPointLayers=None
        self.deq=True
        self.valy=torch.from_numpy(pd.read_csv(fullyPath,index_col=0)['0'].to_numpy()).float().unsqueeze(1)
        self.originy=torch.from_numpy(pd.read_csv(originFullyPath,index_col=0)['0'].to_numpy()).float().unsqueeze(1)
        theta = torch.tensor(np.linspace(0., 180, 360, endpoint=False),dtype=torch.float)
        self.fp=IRadon(512,theta,circle=False,use_filter=HannFilter(),device='cpu')
        self.originImage=torch.from_numpy(pd.read_csv(originImagePath,index_col=0).to_numpy()).float().squeeze()
        plt.ioff()
    

    def training_step(self, batch, batch_idx):
        X,y=batch
        pred=self(X)
        loss=self.lossFunc(pred,y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

    def compare_snr(self,img_test, img_true):
        return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))
        



class CoilwithDEQ(CoilwithDEQBase):
    def __init__(self,
                modelStruct:dict,
                **kwargs):
        super().__init__(**kwargs)
        print (json.dumps(modelStruct, indent=2, default=str))
        numLastOutput=numSkipInput=self.hparams['numInput']*2*self.hparams['L']+self.hparams['numInput']
        self.xEncoder=[]
        self.xEncoderList=[]
        for l in modelStruct['encoder']:
            self.xEncoderList.append(l['type'])
            if l['type']=='fc':
                self.xEncoder.append(LinearBlock(numInput=numLastOutput,numOutput=l['numNeurons'],
                                    batchNorm=self.hparams['norm'] if 'norm' in self.hparams.keys() else False,
                                    acti=l['acti']
                                    ))
            else:
                self.xEncoder.append(LinearBlock(numInput=numLastOutput+numSkipInput,numOutput=l['numNeurons'],
                                    batchNorm=self.hparams['norm'] if 'norm' in self.hparams.keys() else False,
                                    acti=l['acti']
                                    ))
            numLastOutput=l['numNeurons']
        self.xEncoder=nn.ModuleList(self.xEncoder)
        if 'DEQ' in modelStruct.keys():
            self.DEQFixPointLayers=[DEQFixedPoint(DEQ(l['addx'],
                                                    numSkipInput,
                                                    numInput= numLastOutput,
                                                    numEncodeNeurons= l['numNeurons'],
                                                    numExpandLayers= l['numLayers'],
                                                    skipStride= l['skipStride'],
                                                    norm= self.hparams['norm'] if 'norm' in self.hparams.keys() else False,
                                                    acti=l['acti'],
                                                    lastActi=l['lastActi']),
                                                anderson,
                                                tol=1e-4, 
                                                max_iter=50, 
                                                m=5) for l in modelStruct['DEQ']]
            self.DEQFixPointLayers=nn.ModuleList(self.DEQFixPointLayers)
            numLastOutput=numLastOutput*len(modelStruct['DEQ'])
        self.xDecoder=[]
        self.xDecoderList=[]
        if 'extraLayers' in modelStruct['decoder'].keys():
            for l in modelStruct['decoder']['extraLayers']:
                self.xDecoderList.append(l['type'])
                if l['type']=='fc':
                    self.xDecoder.append(LinearBlock(numInput=numLastOutput,numOutput=l['numNeurons'],
                                        batchNorm=self.hparams['norm'] if 'norm' in self.hparams.keys() else False,
                                        acti=l['acti']
                                        ))
                else:
                    self.xDecoder.append(LinearBlock(numInput=numLastOutput+numSkipInput,numOutput=l['numNeurons'],
                                        batchNorm=self.hparams['norm'] if 'norm' in self.hparams.keys() else False,
                                        acti=l['acti']
                                        ))
                numLastOutput=l['numNeurons']
        self.xDecoder.extend([LinearBlock(numLastOutput,numLastOutput//2,
                                    self.hparams['norm'] if 'norm' in self.hparams.keys() else False,acti='relu'),
                                    LinearBlock(numLastOutput//2,numLastOutput//2**2,
                                    self.hparams['norm'] if 'norm' in self.hparams.keys() else False,acti='relu'),
                                    nn.Linear(numLastOutput//2**2,1)])
        if modelStruct['decoder']['sigmoidOutput']:
            self.xDecoder.append(nn.Sigmoid())
        self.xDecoder=nn.ModuleList(self.xDecoder)
    def forward(self,x):
        curr=torch.cat((self.ffm(x),x),dim=1)
        skip=curr.clone().detach()
        for layerName,layer in zip(self.xEncoderList,self.xEncoder):
            if layerName=='fc':
                curr=layer(curr)
            elif layerName=='skip':
                curr=layer(torch.cat((curr,skip),dim=1))
        if self.DEQFixPointLayers is not None:
            DEQResult=[deq(curr,skip,self.deq) for deq in self.DEQFixPointLayers]
            curr=torch.cat(DEQResult,dim=1)
        for layerName,layer in zip(self.xDecoderList,self.xDecoder[0:len(self.xDecoderList)]):
            if layerName=='fc':
                curr=layer(curr)
            elif layerName=='skip':
                curr=layer(torch.cat((curr,skip),dim=1))
        for l in self.xDecoder[len(self.xDecoderList):]:
            curr=l(curr)
        return curr

    def configure_optimizers(self):
        optimizer=optim.AdamW(self.parameters(), lr=self.hparams['lr'])
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": CosineAnnealingLR(optimizer=optimizer,T_max=self.hparams['tmax'],eta_min=1e-6),
                    "monitor": "train_loss",
                    "frequency": 1,
                    "interval": "epoch"
                    }
                }
    def validation_step(self,batch,batch_idx):
        X,_=batch
        pred=self(X)
        return pred.detach().clone().cpu()
    def validation_epoch_end(self,validation_step_outputs):
        #print(f'Validating...\nSize of noisy y label:{self.valy.shape}\nSize of origin y label:{self.originy.shape}\nSize of origin image tensor:{self.originImage.shape}')
        valSino=torch.cat(validation_step_outputs,dim=0)
        if self.hparams['normalized']:
            valSino=valSino*sqrt(2*512**2)
        #print(f'Size of output sinogram:{valSino.shape}')
        snr=self.compare_snr(valSino,self.valy)
        self.logger.experiment.add_scalar('noisy snr',snr,self.current_epoch)
        originSNR=self.compare_snr(valSino,self.originy)
        self.logger.experiment.add_scalar('origin snr',originSNR,self.current_epoch)
        reconImage=self.fp(valSino.reshape((1,1,-1,360))).detach().squeeze()
        #print(f'Size of recon image:{reconImage.shape}')
        imageSNR=self.compare_snr(reconImage,self.originImage)
        self.logger.experiment.add_scalar('Imagesnr',imageSNR,self.current_epoch)
        self.logger.experiment.add_images('image',torch.cat((reconImage.unsqueeze(0),self.originImage.unsqueeze(0)),dim=0).unsqueeze(1).expand(-1,3,-1,-1),self.current_epoch)
        if self.deq==True and self.DEQFixPointLayers is not None:
            self.logger.experiment.add_image("residual", self.getBackwardResPlot(), self.current_epoch)
    def train_dataloader(self):
        trainLoader=DataLoader(DEQDataset(self.hparams['XPath'],self.hparams['yPath']),
                                num_workers=8,pin_memory=True,batch_size=self.hparams['batch_size'],shuffle=True)
        return trainLoader
    def val_dataloader(self):
        valLoader=DataLoader(DEQDataset(self.hparams['fullXPath'],self.hparams['fullyPath']),
                                num_workers=8,pin_memory=True,batch_size=self.hparams['batch_size'],shuffle=False)
        return valLoader
    def getBackwardResPlot(self,dpi=75):
        fig, axs = plt.subplots(len(self.DEQFixPointLayers), 1,figsize=(10, 6), dpi=dpi)
        if len(self.DEQFixPointLayers)==1:
            m=min(self.DEQFixPointLayers[0].backward_res)
            axs.annotate(str(m),(self.DEQFixPointLayers[0].backward_res.index(m),m))
            axs.plot(self.DEQFixPointLayers[0].backward_res)
            
        else:
            for ax,layer in zip(axs,self.DEQFixPointLayers):
                m=min(layer.backward_res)
                axs.annotate(str(m),(layer.backward_res.index(m),m))
                ax.plot(layer.backward_res)
                
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.transpose(img,(2,0,1))

class pictureCoilwithDEQ(CoilwithDEQ):
    def validation_step(self,batch,batch_idx):
        X,_=batch
        pred=self(X)
        pred=pred.detach().cpu().reshape((1,1,256,256))
        self.logger.experiment.add_images('Image',torch.cat((pred,self.originImage.reshape(1,1,256,256)),dim=0).expand(2,3,256,256),self.current_epoch)
        psnr=peak_signal_noise_ratio(pred.numpy(),self.originImage.reshape(1,1,256,256).numpy())
        self.logger.experiment.add_scalar('psnr',psnr,self.current_epoch)
        return
    def validation_epoch_end(self, validation_step_outputs):
        if self.deq==True and self.DEQFixPointLayers is not None:
            self.logger.experiment.add_image("residual", self.getBackwardResPlot(), self.current_epoch)
        return



class DEQDataset(Dataset):
    def __init__(self,XPath,yPath) -> None:
        super().__init__()
        self.X=pd.read_csv(XPath,index_col=0)
        self.y=pd.read_csv(yPath,index_col=0)
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return (torch.from_numpy(self.X.iloc[index].to_numpy()).float(),torch.from_numpy(self.y.iloc[index].to_numpy()).float())
