from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    data_root = './data'
    # 이미 다운로드된 경우엔 download=False
    trainset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    testset  = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)
    return trainset, testset

