import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torch.nn.functional as F
from torchvision.datasets import SVHN
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import functions as f
import classifier as C
import plot as P

transform_one_channel = transforms.Compose([
                                transforms.CenterCrop((28, 28)),
                                transforms.Grayscale(num_output_channels=1),                              
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,],[0.5,])
                                ])

test_dataset=SVHN(root="E:/git/Door_number_finding_system/data",split='test', download=True, transform=transform_one_channel)
train_dataset=SVHN(root="E:/git/Door_number_finding_system/data", download=True, transform=transform_one_channel)

model = C.Classifier()
model=f.to_device(model)
summary(model, (1, 28, 28))

batch_size=100
num_epochs=10

val_percent = 0.2
val_size = int(val_percent * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[train_size,val_size])

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

losses = []
accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
	loss,acc,accuracies,losses=f.train(train_loader,model,accuracies,losses)
	val_loss,val_acc,val_accuracies,val_losses=f.validate(val_loader,model,val_accuracies,val_losses)
			
	print('Epoch [{}/{}] - Loss:{:.4f}, Validation Loss:{:.4f}, Accuracy:{:.2f}, Validation Accuracy:{:.2f}'.format(
		epoch+1, num_epochs, loss.item(), val_loss/100, acc ,val_acc))

test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

P.plot_loss(losses,val_losses)
P.plot_accuracy(accuracies,val_accuracies)

f.evaluate(model,test_loader)