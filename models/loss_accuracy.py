import matplotlib.pyplot as plt

def loss(history):
    print("History : ",history)
    losses = [x['val_loss'] for x in history]
    print("Losses : ",losses)
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.show()

def _accuracy(history):
    print("History : ",history)
    accuracies = [x['val_acc'] for x in history]
    print("Accuracies : ",accuracies)
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()