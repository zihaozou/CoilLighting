import torch
import pytorch_lightning as pl
from DEQLighting import DEQDataset,CoilwithDEQ,LinearBlock,pictureCoilwithDEQ
import torch.nn as nn
import json

jsonWithoutDEQ='model_struct/coilwithoutDEQ.json'
jsonWithDEQ='model_struct/coilwithDEQ.json'
numEpoch=6000


with open(jsonWithDEQ) as jfile:
    ModelStruct=json.load(jfile)


deq=CoilwithDEQ(ModelStruct,lr=1e-3,tmax=numEpoch,norm=False,max_iter=50,L=20)


trainer=pl.Trainer(gpus=1,log_every_n_steps=1,max_epochs=numEpoch,default_root_dir='experiment/Coil',check_val_every_n_epoch=20,num_sanity_val_steps=0)


trainer.fit(deq)


trainer.save_checkpoint('experiment/Coil/CoilWithDEQ.ckpt')

