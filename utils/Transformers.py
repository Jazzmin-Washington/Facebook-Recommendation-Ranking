from torchvision import transforms

def repeat_channel(x):
            return x.repeat(3, 1, 1)

data_transforms = {
            'train':
            transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
         ]),
            'validation': 
            transforms.Compose([
            transforms.Resize(size=512),
            transforms.CenterCrop(size=512),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
    ]),
            'grey':
            transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])  
        }