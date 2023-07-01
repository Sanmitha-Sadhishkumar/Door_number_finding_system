import torch
import torch.nn.functional as F

def loadModel(train_loader,model):
    for images, labels in train_loader:
        outputs = model(images)
        loss = F.cross_entropy(outputs, torch.LongTensor(labels))
        print('Loss:', loss.item())
    print('outputs.shape : ', outputs.shape)
    print('Sample outputs :\n', outputs.data)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, test_loader):
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)

def fit (epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD) :
    _history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        _history.append(result)
    return _history

async def model_val(train_loader,test_loader,model):
    history = [evaluate(model, test_loader)]
    print("epoches : 1e-2")
    history += fit(10,1e-4, model, train_loader, test_loader)
    print("epoches : 1e-3")
    history += fit(10,1e-5, model, train_loader, test_loader)
    return history
