
import torch
import pytorch_lightning as pl
from DEQLighting import DEQDataset,CoilwithDEQ,LinearBlock,pictureCoilwithDEQ
import torch.nn as nn
import json
from tkinter import filedialog as fd


numEpoch=3000


filename = fd.askopenfilename()
with open(filename) as jfile:
    ModelStruct=json.load(jfile)


deq=pictureCoilwithDEQ(ModelStruct,
                lr=.1,
                numInput=2,
                L=20,
                norm=True,
                tmax=numEpoch,
                batch_size=65536,
                XPath='data/imageX.csv',
                yPath='data/imagey.csv',
                fullXPath='data/imageX.csv',
                fullyPath='data/imagey.csv',
                originImagePath='data/01image.csv',)


trainer=pl.Trainer(gpus=1,log_every_n_steps=1,max_epochs=numEpoch,default_root_dir='experiment',check_val_every_n_epoch=5,num_sanity_val_steps=0)


trainer.fit(deq)


trainer.save_checkpoint('experiment/PictureWithDEQ.ckpt')


