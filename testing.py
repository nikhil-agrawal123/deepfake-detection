import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from torch.amp import autocast

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def load_model(model_path, num_classes, device):
    model = models.squeezenet1_0(pretrained=False)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_path, class_names, device):
    image = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        with autocast('cuda',enabled=True):
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]

def check(image_path):
    data_dir = './data'
    batch_size = 32
    _, _, class_names = get_data_loaders(data_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model('squeezenet_model.pth', len(class_names), device)

    image_path = f'{image_path}'  # Replace with the path to your new image
    prediction = predict(model, image_path, class_names, device)

    return prediction