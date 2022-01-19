import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.autograd as autograd
import torch.functional
from math import pi
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import io
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
import torch.optim as optim
class LinearBlock(nn.Module):
    def __init__(self,numInput,numOutput,batchNorm):
        super(LinearBlock,self).__init__()
        modules=[nn.Linear(numInput,numOutput)]
        if batchNorm==True:
            modules.append(nn.BatchNorm1d(numOutput))
        modules.append(nn.LeakyReLU())
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
    def __init__(self,mode,L,numInput,dtype=torch.cuda.FloatTensor):
        super(FFM,self).__init__()
        self.mode=mode
        self.L=L
        stepTensor=torch.arange(end=L,start=0).repeat_interleave(2*numInput)
        index=torch.arange(start=0,end=L*numInput*2,step=numInput*2)
        ffm=torch.empty(L*2*numInput)
        if mode=='linear':
            stepTensor+=1
            for i in range(numInput):
                ffm[index+i]=torch.sin(.5*stepTensor[index]*pi)
                ffm[index+numInput+i]=torch.cos(.5*stepTensor[index]*pi)
        elif mode=='loglinear':
            for i in range(numInput):
                ffm[index+i]=torch.sin(2**stepTensor[index]*pi)
                ffm[index+numInput+i]=torch.cos(2**stepTensor[index]*pi)
        ffm=ffm.unsqueeze(0)
        self.register_buffer('ffmModule',ffm)
    def forward(self,x):
        repeatX=x.repeat(1,self.L*2)
        repeatX=repeatX*self.ffmModule
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
                numskipInput,
                numInput=40,
                numEncodeNeurons=256,
                numExpandLayers=5,
                skipStride=2,
                addz=False,
                norm=False):
        super(DEQ,self).__init__()
        self.expander=LinearBlock(numInput,numEncodeNeurons,norm)
        if skipStride>=numExpandLayers:
            raise ValueError('skipStride should be smaller than numExpandlayers')
        self.skipList=list(range(skipStride,numExpandLayers,skipStride+1))
        expandLst=[]
        for i in range(numExpandLayers):
            if i in self.skipList:
                expandLst.append(LinearBlock(numEncodeNeurons+numskipInput,numEncodeNeurons,norm))
            else:
                expandLst.append(LinearBlock(numEncodeNeurons,numEncodeNeurons,norm))
        self.expandLayers=nn.ModuleList(expandLst)
        self.decoder=nn.ModuleList([LinearBlock(numEncodeNeurons,numInput,norm),
                                    LinearBlock(numInput,numInput,norm),
                                    LinearBlock(numInput,numInput,norm)]
                                    )
        self.skipStride=skipStride
        self.addz=addz
        
    def forward(self,z,x,skip,deq):
        if deq:
            curr=self.expander(z)
            for i,l in enumerate(self.expandLayers):
                if i in self.skipList:
                    curr=l(torch.cat((curr,skip),dim=1))
                else:
                    curr=l(curr)
            for i,l in enumerate(self.decoder):
                if i==1:
                    curr=l(curr+x)
                elif self.addz==True and i==2:
                    curr=l(curr+z)
                else:
                    curr=l(curr)
        else:
            curr=self.expander(x)
            for i,l in enumerate(self.expandLayers):
                if i in self.skipList:
                    curr=l(torch.cat((curr,skip),dim=1))
                else:
                    curr=l(curr)
            for l in self.decoder:
                curr=l(curr)
        return curr

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver,**kwargs):
        super(DEQFixedPoint,self).__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
    def forward(self, x,skip,deq):
        # compute forward pass and re-engage autograd tape
        if deq:
            with torch.no_grad():
                z, self.forward_res = self.solver(lambda z : self.f(z, x,skip,deq), torch.zeros_like(x), **self.kwargs)
            newz = self.f(z.requires_grad_(),x,skip,deq)
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
            newz=self.f(None, x,skip,deq)
        return newz
    




class CoilwithDEQBase(pl.LightningModule):
    def __init__(self,
                lr=.1,
                batch_size=65251,
                L=10,
                numInput=2,
                showEvery=20
                ):
        super().__init__()
        self.ffm=FFM('loglinear',L,numInput)
        self.mse=nn.MSELoss()
        self.batch_size=batch_size
        self.lr=lr
        self.showEvery=showEvery
        self.epoch=0
        self.DEQFixPointLayers=None
        self.deq=True
        plt.ioff()
    

    def training_step(self, batch, batch_idx):
        X,y=batch
        pred=self(X)
        loss=self.mse(pred,y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
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
    def on_epoch_end(self) -> None:
        self.epoch+=1
        if self.epoch%self.showEvery==0 and self.deq==True:
            self.logger.experiment.add_image("residual", self.getBackwardResPlot(), self.epoch)
    def train_dataloader(self):
        trainLoader=DataLoader(DEQDataset(),num_workers=3,pin_memory=True,batch_size=self.batch_size,shuffle=True)
        return trainLoader





class CoilwithDEQ(CoilwithDEQBase):
    def __init__(self,
                xEncoderLst=[],
                numDEQ=1,
                numEncodeNeurons=128,
                numDEQNeurons=256,
                addz=True,
                norm=True,
                numExpandLayer=3,
                skipStride=2,
                **kwargs):
        super().__init__(**kwargs)
        numInput=kwargs['numInput']
        L=kwargs['L']
        lastNumOutput=numInput*L*2
        self.xEncoder=[]
        for l in xEncoderLst:
            if l=='fc':
                self.xEncoder.append(LinearBlock(lastNumOutput,numEncodeNeurons,norm))
            elif l=='skip':
                self.xEncoder.append(LinearBlock(lastNumOutput+numInput*L*2,numEncodeNeurons,norm))
            lastNumOutput=numEncodeNeurons
        self.xEncoder=nn.ModuleList(self.xEncoder)
        self.xEncoderLst=xEncoderLst
        self.DEQFixPointLayers=[DEQFixedPoint(DEQ(numInput*L*2,
                                                numInput= lastNumOutput,
                                                numEncodeNeurons= numDEQNeurons[i] if isinstance(numDEQNeurons,list) else numDEQNeurons,
                                                numExpandLayers= numExpandLayer[i] if isinstance(numExpandLayer,list) else numExpandLayer,
                                                skipStride= skipStride[i] if isinstance(skipStride,list) else skipStride,
                                                addz=addz[i] if isinstance(addz,list) else addz,
                                                norm= norm[i] if isinstance(norm,list) else norm),
                                            anderson,
                                            tol=1e-4, 
                                            max_iter=50, 
                                            m=5) for i in range(numDEQ)]
        self.DEQFixPointLayers=nn.ModuleList(self.DEQFixPointLayers)
        self.decoder=nn.Sequential(LinearBlock(lastNumOutput*numDEQ,lastNumOutput*numDEQ//2,norm),
                                    LinearBlock(lastNumOutput*numDEQ//2,numEncodeNeurons//2**2,norm),
                                    nn.Linear(numEncodeNeurons//2**2,1))
    def forward(self,x):
        curr=self.ffm(x)
        skip=curr.clone().detach()
        for layerName,layer in zip(self.xEncoderLst,self.xEncoder):
            if layerName=='fc':
                curr=layer(curr)
            elif layerName=='skip':
                curr=layer(torch.cat((curr,skip),dim=1))
        DEQResult=[deq(curr,skip,self.deq) for deq in self.DEQFixPointLayers]
        curr=torch.cat(DEQResult,dim=1)
        return self.decoder(curr)

    def configure_optimizers(self):
        optimizer=optim.AdamW(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": CosineAnnealingLR(optimizer=optimizer,T_max=2000,eta_min=1e-6),
                    "monitor": "train_loss",
                    "frequency": 1,
                    "interval": "epoch"
                    }
                }




class DEQDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X=pd.read_csv('X.csv',header=None)
        self.y=pd.read_csv('y.csv',header=None)
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return (torch.from_numpy(self.X.iloc[index].to_numpy()).float(),torch.from_numpy(self.y.iloc[index].to_numpy()).float())
