import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import os


x_train_padding = np.load('./Data/X_train_padding.npy')
y_train_padding = np.load('./Data/y_train_padding.npy')

print(f"the shape of the features {x_train_padding.shape}")
print(f"the shape of the target {y_train_padding.shape}")

x_train, x_val, y_train, y_val = train_test_split(x_train_padding, y_train_padding, test_size=0.01, random_state=42)

x_train = torch.from_numpy(x_train).float().permute(0,2,1)
y_train = torch.from_numpy(y_train).float().permute(0,2,1)
x_val = torch.from_numpy(x_val).float().permute(0,2,1)
y_val = torch.from_numpy(y_val).float().permute(0,2,1)


class Model(nn.Module):
    def __init__(self,num_in,num_out,seq_length):
        super().__init__()
        self.model = nn.Sequential(nn.Conv1d(in_channels=12, out_channels=64, kernel_size=3,padding='same'),
                                nn.LayerNorm((64,seq_length)),
                                nn.GELU(approximate='tanh'),
                                nn.Dropout(0.7),
                                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding='same'),
                                nn.LayerNorm((128,seq_length)),
                                nn.GELU(approximate='tanh'),
                                nn.Dropout(0.7),
                                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3 ,padding='same'),
                                nn.LayerNorm((256,seq_length)),
                                nn.GELU(approximate='tanh'),
                                nn.Dropout(0.7),
                                nn.AdaptiveAvgPool1d(1),                              
                                )
        
        self.linear = nn.Sequential(nn.Linear(256,128),
                        nn.GELU(approximate='tanh'),
                        nn.Dropout(0.6),
                        nn.Linear(128,64),
                        nn.GELU(approximate='tanh'),
                        nn.Dropout(0.6),
                        nn.Linear(64,num_out)
                        )
        
        self.apply(self._init_weights)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")
    def forward(self,x):
        x=self.model(x)
        x=self.linear(x.view(1,x.shape[0]))
        return x.view(14,12246)  

    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0 , std=0.02)
            

num_epochs = 10
batch_size = x_train.shape[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Used device {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    

def validate():
    model.eval()
    loss_accum=0.0
    with torch.no_grad():
        for i in range(x_val.shape[0]):
            y_pred = model(x_val[i].to(device))
            loss = loss_fn(y_pred,y_val[i].to(device))
            loss_accum += loss.detach()
    loss_accum /= x_val.shape[0]
    return loss_accum.item()


max_lr= 0.080
min_lr= 0.0060
warmup_steps = 1

def get_lr(it):
    # 1) if linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    # 2) if it > lr_decay_iters return min_lr
    if it > num_epochs:
        return min_lr
    # 3) in between , use cosine decay down to min lr
    decay_ratio = ((it - warmup_steps) / (num_epochs - warmup_steps) ) **0.2
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


model = Model(x_train.shape[1],y_train.shape[1]*y_train.shape[-1],x_train.shape[-1]).to(device)


optimizer = torch.optim.Adagrad(model.parameters(),lr=max_lr)
loss_fn = nn.MSELoss()

log_dir = 'EMG/Logs'
os.makedirs(log_dir,exist_ok=True)
log_file = os.path.join(log_dir,f'CNN_Sequential_log.txt')
with open(log_file,'w') as f:
    pass

train_lossi=[]
val_lossi=[]
for epoch in range(num_epochs):
    model.train()
    loss_acum=0.0
    optimizer.zero_grad()

    for batch in range(batch_size):
        x_train_batch , y_train_batch = x_train[batch],y_train[batch]

        x_train_batch = x_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)

        #forward pass
        y_pred = model(x_train_batch)
        loss = loss_fn(y_pred,y_train_batch)
        #backward pass
        loss = loss / batch_size
        loss.backward()
        #accumulated loss
        loss_acum += loss.detach()

    norm = nn.utils.clip_grad_norm_(model.parameters() , 1.0)
    #update
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

    optimizer.step()
    val_loss = validate()
    val_lossi.append(val_loss)
    print(f'epoch: {epoch+1} |  loss: {loss_acum.item():.4f} |  norm: {norm:.3f} | lr:{lr:.3f} | val loss:{val_loss:.4f}')
    train_lossi.append(loss_acum.item())
    with open(log_file,'a') as f:
            f.write(f'epoch: {epoch+1} |  MSE_loss: {loss_acum.item():.2f} | norm: {norm:.2f} | lr:{lr:.3f} | val loss:{val_loss:.2f}\n')
    

plt.plot(train_lossi)
plt.plot(val_lossi)
plt.legend(['train','val'])
plt.savefig("CNN_Sequential_plot.png")
plt.show()
