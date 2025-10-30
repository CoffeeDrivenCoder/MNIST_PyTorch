from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as tsf
import cv2


batch_size = 64
transform = tsf.Compose([tsf.ToTensor(), tsf.Normalize([0.1307], [0.3081])])
# Normalize:正则化，降低模型复杂度,防止过拟合


train_set=datasets.MNIST(root="data",train=True,download=True,transform=transform)
test_set=datasets.MNIST(root="data",train=False,download=True,transform=transform)


def get_data_loader():
    train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)
    return train_loader,test_loader











