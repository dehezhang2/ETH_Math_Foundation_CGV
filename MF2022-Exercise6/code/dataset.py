from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os

class SRDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): High resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(self, image_dir, image_size = 64, upscale_factor = 2, jitter_val = 0.2, mode = 'Train', interpolation = InterpolationMode.BILINEAR) -> None:
        super(SRDataset, self).__init__()
        self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in os.listdir(image_dir)]
        self.image_size = image_size
        self.upscale_factor = upscale_factor
        self.jitter_val = jitter_val
        self.mode = mode
        self.interpolation = interpolation

    def __getitem__(self, index):
        image = read_image(self.image_file_names[index]).float()/255.
        if self.mode == 'Train':
            hr_transformer = transforms.Compose([
                transforms.RandomCrop(self.image_size),
                transforms.ColorJitter(brightness=self.jitter_val, contrast=self.jitter_val, saturation=self.jitter_val, hue=self.jitter_val),
            ])  
        elif self.mode == 'Test':
            hr_transformer = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.ColorJitter(brightness=self.jitter_val, contrast=self.jitter_val, saturation=self.jitter_val, hue=self.jitter_val),
            ])
        elif self.mode == 'Test_both':
            hr_transformer = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.ColorJitter(brightness=self.jitter_val, contrast=self.jitter_val, saturation=self.jitter_val, hue=self.jitter_val),
            ])
            lr_transformer1 = transforms.Compose([
                transforms.Resize(size=(int(self.image_size / self.upscale_factor), int(self.image_size / self.upscale_factor)), interpolation = InterpolationMode.BILINEAR),
            ])
            lr_transformer2 = transforms.Compose([
                transforms.Resize(size=(int(self.image_size / self.upscale_factor), int(self.image_size / self.upscale_factor)), interpolation = InterpolationMode.BICUBIC),
            ])
            lr_transformer3 = transforms.Compose([
                transforms.Resize(size=(int(self.image_size / self.upscale_factor), int(self.image_size / self.upscale_factor)), interpolation = InterpolationMode.NEAREST),
            ])
            hr_image = hr_transformer(image)
            lr_image1 = lr_transformer1(hr_image)
            lr_image2 = lr_transformer2(hr_image)
            lr_image3 = lr_transformer3(hr_image)
            return lr_image1, lr_image2, lr_image3, hr_image
        
        lr_transformer = transforms.Compose([
            transforms.Resize(size=(int(self.image_size / self.upscale_factor), int(self.image_size / self.upscale_factor)), interpolation = self.interpolation),
        ])
        hr_image = hr_transformer(image)
        lr_image = lr_transformer(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_file_names)
