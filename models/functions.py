import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data.dataloader import DataLoader

criterion = nn.CrossEntropyLoss()

def test_one(model,src):
	batch_size=1
	img,label=DataLoader(srcbatch_size=batch_size,shuffle=True,pin_memory=True)
	images=to_device(images)
	outputs = model(images)
	_, predicted = torch.max(outputs.data, 1)
	return predicted

def train(train_loader,model,accuracies,losses):
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	for i, (images, labels) in enumerate(train_loader):
		# Forward pass
		images=to_device(images)
		labels=to_device(labels)
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
	return loss,acc,accuracies,losses
		
def validate(val_loader,model,val_accuracies,val_losses):
	val_loss = 0.0
	val_acc = 0.0
	with torch.no_grad():
		for images, labels in val_loader:
			images=to_device(images)
			labels=to_device(labels)
			outputs = model(images)
			loss = criterion(outputs, labels)
			val_loss += loss.item()
			
			_, predicted = torch.max(outputs.data, 1)
		total = labels.size(0)
		correct = (predicted == labels).sum().item()
		val_acc += correct / total
		val_accuracies.append(val_acc)
		val_losses.append(loss.item())
		return val_loss,val_acc,val_accuracies,val_losses
	

def to_device(a):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	a.to(device)
	return a

def evaluate(model,test_loader):
	model.eval()

	with torch.no_grad():
		correct = 0
		total = 0
		y_true = []
		y_pred = []
		for images, labels in test_loader:
			images = to_device(images)
			labels = to_device(labels)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			predicted=predicted.to('cpu')
			labels=labels.to('cpu')
			y_true.extend(labels)
			y_pred.extend(predicted)

	print('\nTest Accuracy: {}%'.format(100 * correct / total))

	print(classification_report(y_true, y_pred))
