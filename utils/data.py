import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
root = os.path.join('/home', os.environ.get('USER'), 'datasets')

def get_MNIST(batch_size):
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(root, 'mnist'),
                train=True, download=True,
                transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(root, 'mnist'),
                train=False, download=True,
                transform=transforms.ToTensor()),
            batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def get_FashionMNIST(batch_size):
    train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(os.path.join(root, 'fashion_mnist'),
                train=True, download=True,
                transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(os.path.join(root, 'fashion_mnist'),
                train=False, download=True,
                transform=transforms.ToTensor()),
            batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def get_CIFAR10(batch_size):
    normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(root, 'cifar10'),
                train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(root, 'cifar10'),
                train=False, download=True, transform=transform),
            batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def get_CIFAR100(batch_size):
    normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(root, 'cifar100'),
                train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(root, 'cifar100'),
                train=False, download=True, transform=transform),
            batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, test_loader
