import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report



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
