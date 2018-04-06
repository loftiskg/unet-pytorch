import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
import glob
from sklearn.preprocessing import minmax_scale
from torchvision import transforms

INPUT_DATA_FOLDER = './sample_data/input'
TARGET_DATA_FOLDER = './sample_data/target'
class SampleDataDriver(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        self.__xs = glob.glob(os.path.join(INPUT_DATA_FOLDER, '*.png'))
        self.__ys = glob.glob(os.path.join(TARGET_DATA_FOLDER, '*.png'))

    def __len__(self):
        return len(self.__xs)

    def __getitem__(self, idx):
        input = Image.open(self.__xs[idx])
        target = Image.open(self.__ys[idx])

        input = np.array(input)
        target = np.array(target)

        input = input.reshape(1, input.shape[0], input.shape[1])
        target = target.reshape(1, target.shape[0], target.shape[1])


        data = {'input': input, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data

class TwoClassOutput(object):
    def __call__(self, sample):
        input, target = sample['input'], sample['target']


class ToTensor(object):
    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        return {'input': torch.from_numpy(input), 'target': torch.from_numpy(target)}

class Normalize(object):
    def __call__(self, sample):
        input, target = (sample['input']/255) - 0.5, (sample['target']/255) - 0.5
        return {'input': input, 'target': target}

if __name__ == '__main__':

    transform = transforms.Compose([Normalize()])
    data = SampleDataDriver(transform=transform)

    input = data[0]['input']
    target = data[0]['target']

    print(input)
    print(target)

    input = np.squeeze(input)
    target = np.squeeze(target)

    input = minmax_scale(input, feature_range=(0, 255))
    target = minmax_scale(target, feature_range=(0, 255))

    print(input)
    print(target)

    input = Image.fromarray(input)
    target = Image.fromarray(target)

    input.show()
    target.show()