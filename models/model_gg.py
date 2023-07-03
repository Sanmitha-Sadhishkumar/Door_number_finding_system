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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size=100
num_epochs=10

val_percent = 0.2
val_size = int(val_percent * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[train_size,val_size])

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

losses = []
accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# Forward pass
		images=f.to_device(images)
		labels=f.to_device(labels)
		outputs = model(images)
		loss = criterion(outputs, labels)
		
		# Backward pass and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		_, predicted = torch.max(outputs.data, 1)
	acc = (predicted == labels).sum().item() / labels.size(0)
	accuracies.append(acc)
	losses.append(loss.item())
		
	# Evaluate the model on the validation set
	val_loss = 0.0
	val_acc = 0.0
	with torch.no_grad():
		for images, labels in val_loader:
			images=f.to_device(images)
			labels=f.to_device(labels)
			outputs = model(images)
			loss = criterion(outputs, labels)
			val_loss += loss.item()
			
			_, predicted = torch.max(outputs.data, 1)
		total = labels.size(0)
		correct = (predicted == labels).sum().item()
		val_acc += correct / total
		val_accuracies.append(acc)
		val_losses.append(loss.item())
	
			
	print('Epoch [{}/{}],Loss:{:.4f},Validation Loss:{:.4f},Accuracy:{:.2f},Validation Accuracy:{:.2f}'.format(
		epoch+1, num_epochs, loss.item(), val_loss, acc ,val_acc))


test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

P.plot_loss(losses,val_losses)
P.plot_accuracy(accuracies,val_accuracies)

f.evaluate(model,test_loader)

