import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

bs = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=bs,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=bs
)

for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break

model = models.MyModel()

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 0.003
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.3f}%')

n_epochs = 5


loss_train_histry = []
loss_test_histry = []
acc_train_histry = []
acc_test_histry = []

for k in range(n_epochs):
    print(f'epoch {k+1}/{n_epochs}', end=':', flush=True)

    time_start = time.time()
    loss_train = models.train(model, dataloader_train, loss_fn, optimizer)
    time_end = time.time()
    loss_train_histry.append(loss_train)
    print(f'train loss: {loss_train:3f} ({time_end-time_start}s)')
    print(f'train loss: {loss_train:3f} ({time_end-time_start:.1f}s)')


    loss_test = models.test(model, dataloader_test, loss_fn)
    loss_test_histry.append(loss_test)
    print(f'test loss: {loss_test:3f}', end=', ')

    acc_train = models.test_accuracy(model, dataloader_train)
    acc_train_histry.append(acc_train)
    print(f'train accuracy: {acc_train*100:.3f}%', end=', ')

    acc_test = models.test_accuracy(model, dataloader_test)
    acc_test_histry.append(acc_train)
    print(f'test accuracy: {acc_test*100:.2f}%')


    plt.plot(acc_train_histry, label='train')
    plt.plot(acc_test_histry, label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(loss_train_histry, label='train')
    plt.plot(loss_test_histry, label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.show()







#models.train(model, dataloader_train, loss_fn, optimizer)


