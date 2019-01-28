import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import CifarDataLoader
from model.vgg_net import VGGNet

# global accuracy
best_accuracy = 0.0

# choose the device gpu / cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load cifar10 data.
data_loader = CifarDataLoader('./data/cifar-10-batches-py/', 'data_batch_', 'test_batch')
data_loader.load_data()
train_loader = data_loader.get_data_loader(batch_size=100, shuffle=True)
test_loader = data_loader.get_data_loader(test=True)

# Load model
net = VGGNet()
net = net.to(device)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print("Batch %d: Train Loss is %.3f and Accuracy is %.3f (%d/%d)" % (batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    global best_accuracy
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print("Test Loss is %.3f and Accuracy is %.3f (%d/%d)" % (test_loss / len(test_loader), 100. * correct / total, correct, total))
    acc = 100.*correct/total
    if acc > best_accuracy:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_accuracy = acc

for epoch in range(100):
    train(epoch)
    test(epoch)