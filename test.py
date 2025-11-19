import os
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as tvu
from tqdm import tqdm

def save_cifar10_images(root="data/cifar10"):
    transform = transforms.ToTensor()
    
    # ----------------------------
    # Train set
    # ----------------------------
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    train_folder = os.path.join(root, "train")
    os.makedirs(train_folder, exist_ok=True)
    
    for i, (img, label) in enumerate(tqdm(trainset, desc="Saving train images")):
        tvu.save_image(img, os.path.join(train_folder, f"{i}.png"))
    
    # ----------------------------
    # Test set
    # ----------------------------
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    test_folder = os.path.join(root, "test")
    os.makedirs(test_folder, exist_ok=True)
    
    for i, (img, label) in enumerate(tqdm(testset, desc="Saving test images")):
        tvu.save_image(img, os.path.join(test_folder, f"{i}.png"))

    print("Done! CIFAR-10 images saved.")
    
save_cifar10_images()