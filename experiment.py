
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


deq=pictureCoilwithDEQ.load_from_checkpoint('experiment/PictureWithDEQ.ckpt',lr=5e-3)


trainer=pl.Trainer(gpus=1,log_every_n_steps=1,max_epochs=numEpoch,default_root_dir='experiment',check_val_every_n_epoch=5,num_sanity_val_steps=0)


trainer.fit(deq)
 

trainer.save_checkpoint('experiment/PictureWithDEQ.ckpt')

