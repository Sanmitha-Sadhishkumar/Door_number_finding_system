import torch
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import svhn_class as S
import asyncio
import svhn_func as SF
import loss_accuracy as LA

test_ds=SVHN(root="E:/git/Door_number_finding_system/data",split='test', download=True, transform=ToTensor())
train_ds=SVHN(root="E:/git/Door_number_finding_system/data", download=True, transform=ToTensor())
history=[]

batch_size=64
train_loader=DataLoader(train_ds,batch_size,shuffle=True,num_workers=4,pin_memory=True)
test_loader=DataLoader(test_ds,batch_size,num_workers=4,pin_memory=True)

input_size = 3072
num_classes = 10   

model = S.SvhnModel(input_size, out_size=num_classes)

if __name__=='__main__':
    torch.multiprocessing.freeze_support()
    SF.loadModel(train_loader,model)
    torch.multiprocessing.freeze_support()
    history = asyncio.run(SF.model_val(train_loader,test_loader,model))
    LA.loss(history)
    LA._accuracy(history)