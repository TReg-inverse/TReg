from glob import glob
from typing import Any, Callable, Optional, Tuple
from pathlib import Path

from munch import munchify
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101, VisionDataset

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: str,
                   root: str,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool,
                   rescaled: bool,
                   **kwargs):

    # tf = [transforms.ToTensor()]
    tf = [
            transforms.Resize(512),
            transforms.ToTensor()
        ]
    if rescaled:
        tf.append(transforms.Normalize((0.5, 0.5, 0.5),
                                       (0.5, 0.5, 0.5)))
    tf = transforms.Compose(tf)

    dataset = get_dataset(name=dataset,
                          root=root,
                          transform=tf,
                          **kwargs)

    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='food')
class Food101Dataset(Food101):
    def __init__(self, root: str, transform: Optional[Callable]=None, **kwargs):
        super().__init__(root, transform=transform, download=True, split='test')

        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ])

        # import ipdb; ipdb.set_trace()
        if kwargs.get('target') is not None:
            target_label = self.class_to_idx[kwargs['target']]

            images = []
            for (img, label) in zip(self._image_files, self._labels):
                if label == target_label:
                    images.append(img)
            self._image_files = images
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img,_= super().__getitem__(idx)
        return img
    
@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transform: Optional[Callable]=None, **kwargs):
        super().__init__(root, transform=transform)

        # self.fpaths = sorted(glob(root + '/**/*.png', recursive=True)) + sorted(ro
        self.fpaths = sorted(root.glob('*.png')) + sorted(root.glob('*.jpg'))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img

@register_dataset(name='landscape')
class LandDataset(FFHQDataset):
    def __init__(self, root, transform, **kwargs):
        super().__init__(root, transform=transform)
    
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()
        ])


@register_dataset(name='imagenet1k_i2sb')
class ImageNet1kI2SBDataset(FFHQDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, **kwargs):

        with open('exp/data_category/imagenet_val.txt', 'r') as f:
            fpaths = f.readlines()
        self.fpaths = [fname.strip() for fname in fpaths]
        self.fpaths = self.fpaths[::10]

        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index: int):
        return super().__getitem__(index)


@register_dataset(name='imagenet1k')
class ImageNet1kDataset(FFHQDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, **kwargs):
        super().__init__(root, transform=transform)
        
        self.fpaths = self.fpaths[:1000]  # only takes the first 1k images
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index: int):
        return super().__getitem__(index)

@register_dataset(name='afhq')
class AFHQDataset(VisionDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, **kwargs):
        super().__init__(root, transform=transform)

        # self.fpaths = sorted(glob(root + '/**/*.jpg', recursive=True))
        self.fpaths = sorted(list(root.glob('**/*.jpg')))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img

@register_dataset(name='imagenet_pair')
class ImageNetPair(VisionDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, **kwargs):
        super().__init__(root, transform=transform)

        self.root = Path(root)

        # construct class dictionary key:value = dirname:class (e.g. n07730033:cardoon)
        with open('imagenet_pair/imagenet_class.txt', 'r') as f:
            val_class = [x.strip().replace("{","").replace("}","").replace(",","") for x in f.readlines()] 
        val_class = [x.split(": ") for x in val_class]
        self.val_class = {x[1].replace("'", ""): x[0].replace("'", "") for x in val_class}

        # construct wordnet class pair
        with open('imagenet_pair/wordnet.is_a.txt', 'r') as f:
            pairs = [x.strip() for x in f.readlines()]
        pairs = [x.split(" ") for x in pairs]
        self.pairs = {x[0]: x[1] for x in pairs}
        self.inv_pairs = {x[1]: x[0] for x in pairs}

        # Get image paths
        with open('imagenet_pair/imagenet_val.txt', 'r') as f:
            fpaths = [x.strip() for x in f.readlines()]

        # take one sample from each class
        self.fpaths = fpaths[::10]

        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ])

    def get_dirname(self, path):
        return path.split('/')[0]

    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        dirname = self.get_dirname(fpath)
        fpath = self.root.joinpath(fpath)

        orig_class = self.val_class[dirname]
        import ipdb; ipdb.set_trace()
        if self.pairs.get(dirname, None) is None:
            target_class = self.val_class[self.inv_pairs[dirname]]
        else:
            target_class = self.val_class[self.pairs[dirname]]

        img = Image.open(fpath).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        return img, orig_class, target_class