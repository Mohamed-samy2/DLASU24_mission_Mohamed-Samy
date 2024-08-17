import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



x_train_tabular = np.load('./Data/X_train_tabular.npy')
y_train_tabular = np.load('./Data/y_train_tabular.npy')

print(f"the shape of the features {x_train_tabular.shape}")
print(f"the shape of the target {y_train_tabular.shape}")


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_tabular, y_train_tabular, test_size=0.015, random_state=42)

print(f"Number of Training examples {x_train.shape[0]}")
print(f"Number of Validation examples {x_val.shape[0]}")

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).float()

class Model(nn.Module):
    def __init__(self,num_in,num_out):
        super().__init__()
        
        self.model = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3,padding='same'),
                                nn.LayerNorm((32 , num_in)),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3 ,padding='same'),
                                nn.LayerNorm((64,num_in)),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3 ,padding='same'),
                                nn.LayerNorm((128,num_in)),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Flatten(),
                                nn.Linear(128*num_in,num_out)
                                )
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")
        
        self.apply(self._init_weights)
    def forward(self,x):
        x=x.unsqueeze(1)
        return self.model(x)

    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0 , std=0.02)


num_epochs = 10
batch_size = 2048
total_batch_size = 2**22
device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert total_batch_size % batch_size == 0 ," make sure total_batch_size divisble by batch_size"
grad_accum_steps = total_batch_size // batch_size
print(f"Total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumlation steps : {grad_accum_steps}")
print(f"Used device {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

max_lr= 0.1
min_lr= 0.0020
warmup_steps = 1

def get_lr(it):
    # 1) if linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    # 2) if it > lr_decay_iters return min_lr
    if it > num_epochs:
        return min_lr
    # 3) in between , use cosine decay down to min lr
    decay_ratio = ((it - warmup_steps) / (num_epochs - warmup_steps) ) ** 0.2
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def validate():
    model.eval()
    with torch.no_grad():
        y_pred = model(x_val.to(device))
        loss = loss_fn(y_pred,y_val.to(device))
    return loss.item()

def batch_generator():
    num_samples = x_train.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield x_train[batch_indices], y_train[batch_indices]

model = Model(x_train.shape[1],y_train.shape[1]).to(device)

optimizer = torch.optim.Adagrad(model.parameters(),lr=max_lr)
loss_fn = nn.MSELoss()

train_lossi=[]
val_lossi=[]
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    loss_acum=0.0
    train_gen = batch_generator()
    
    for acc in range(grad_accum_steps):
    
        x_train_batch , y_train_batch = next(train_gen)
        x_train_batch = x_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)

        #forward pass
        y_pred = model(x_train_batch)
        loss = loss_fn(y_pred,y_train_batch)
        loss = loss / grad_accum_steps
        loss_acum += loss.detach()
        #backward pass
        loss.backward()
    
    norm = nn.utils.clip_grad_norm_(model.parameters() , 1.0)

    #update
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
        
    optimizer.step()
    val_loss = validate()
    val_lossi.append(val_loss)
    
    print(f'epoch: {epoch+1} |  MSE_loss: {loss_acum.item():.2f} | norm: {norm:.2f} | lr:{lr:.3f} | val loss:{val_loss:.2f}')
    train_lossi.append(loss_acum.item())

plt.plot(train_lossi)
plt.plot(val_lossi)
plt.legend(['train','val'])
plt.title("Training and Validation Loss")
plt.savefig('CNN_tabular_plot.png')
plt.show()

