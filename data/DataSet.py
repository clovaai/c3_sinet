import torch
import cv2
import torch.utils.data
from PIL import Image



class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, imList, labelList, transform=None):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe CVTransforms.py for supported transformations
        '''
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        if self.transform:
            [image, label] = self.transform(image, label)
        return (image, label)

class PILDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, imList, labelList, transform=None, Double=False):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe CVTransforms.py for supported transformations
        '''
        self.imList = imList
        self.labelList = labelList
        self.transform = transform
        self.Double = Double

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image =  Image.open(image_name).convert('RGB')
        label = Image.open(label_name)

        if self.Double:
            if self.transform:
                [image, label_coarse, label] = self.transform(image, label)
            return (image, label_coarse, label)
        else:
            if self.transform:
                [image, label] = self.transform(image, label)
            return (image, label)
