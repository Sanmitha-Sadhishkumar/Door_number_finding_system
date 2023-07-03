import matplotlib.pyplot as plt
num_epochs=10
def plot_loss(losses,val_losses):
	plt.plot(range(num_epochs),
		losses, color='red',
		label='Training Loss',
		marker='o')
	plt.plot(range(num_epochs),
		val_losses,
		color='blue',
		linestyle='--',
		label='Validation Loss',
		marker='x')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training and Validation Loss')
	plt.legend()
	plt.show()

def plot_accuracy(accuracies,val_accuracies):
	plt.plot(range(num_epochs),
		accuracies,
		label='Training Accuracy',
		color='red',
		marker='o')
	plt.plot(range(num_epochs),
		val_accuracies,
		label='Validation Accuracy',
		color='blue',
		linestyle=':',
		marker='x')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Training and Validation Accuracy')
	plt.legend()
	plt.show()