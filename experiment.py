# %%
import torch
import pytorch_lightning as pl
from DEQLighting import DEQDataset,CoilwithDEQ,LinearBlock
import torch.nn as nn
import json

# %%
numEpoch=800

# %%
with open('modelStruct.json') as jFile:
    unnormalizedModelStruct=json.load(jFile)
with open('model_normalized.json') as jFile:
    normalizedModelStruct=json.load(jFile)


# %%
deq=CoilwithDEQ(unnormalizedModelStruct,
                lr=.1,
                numInput=2,
                numEncodeNeurons=256,
                addz=False,
                showEvery=1,
                L=20,
                norm=True,
                tmax=numEpoch)

# %%
trainer=pl.Trainer(gpus=1,log_every_n_steps=1,max_epochs=numEpoch,default_root_dir='experiment',check_val_every_n_epoch=20,num_sanity_val_steps=0)

# %%
trainer.fit(deq)

# %%
trainer.save_checkpoint('experiment/9layers_unnormalized.ckpt')


