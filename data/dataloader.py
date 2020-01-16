import os
import data.CVTransforms as cvTransforms
import data.PILTransform as pilTransforms
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms
import data.DataSet as myDataLoader
import torch
import data.loadData as ld
import pickle

def cityPIL_Doublerandscalecrop( cached_data_file, data_dir, classes, batch_size, num_work=6,
                 scale=(0.5, 2.0), size=(1024, 512), scale1 = 1, scale2 = 2, ignore_idx=255):
    print("This input size is  " +str(size))

    if not os.path.isfile(cached_data_file):
        dataLoad = ld.LoadData(data_dir, classes, cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(cached_data_file, "rb"))

    if isinstance(size, tuple):
        size = size
    else:
        size = (size, size)

    if isinstance(scale, tuple):
        scale = scale
    else:
        scale = (scale, scale)


    train_transforms = pilTransforms.Compose(
        [
            # pilTransforms.data_aug_color(),
            pilTransforms.RandomScale(scale=scale),
            pilTransforms.RandomCrop(crop_size=size,ignore_idx=ignore_idx),
            pilTransforms.RandomFlip(),
            pilTransforms.DoubleNormalize(scale1=scale1, scale2=scale2)
        ]
    )
    val_transforms = pilTransforms.Compose(
        [
            pilTransforms.Resize(size=size),
            pilTransforms.DoubleNormalize(scale1=scale2, scale2=1)
        ]
    )
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.PILDataset(data['trainIm'], data['trainAnnot'], Double=True, transform=train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.PILDataset(data['valIm'], data['valAnnot'], Double=True, transform=val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=num_work, pin_memory=True)

    return trainLoader, valLoader, data

def cityPIL_randscalecrop( cached_data_file, data_dir, classes, batch_size, num_work=6,
                 scale=(0.5, 2.0), size=(1024, 512), scale1 = 1, ignore_idx=255):
    print("This input size is  " +str(size))

    if not os.path.isfile(cached_data_file):
        dataLoad = ld.LoadData(data_dir, classes, cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(cached_data_file, "rb"))

    if isinstance(size, tuple):
        size = size
    else:
        size = (size, size)

    if isinstance(scale, tuple):
        scale = scale
    else:
        scale = (scale, scale)


    train_transforms = pilTransforms.Compose(
        [
            pilTransforms.RandomScale(scale=scale),
            pilTransforms.RandomCrop(crop_size=size,ignore_idx=ignore_idx),
            pilTransforms.RandomFlip(),
            pilTransforms.Normalize(scaleIn=scale1)
        ]
    )
    val_transforms = pilTransforms.Compose(
        [
            pilTransforms.Resize(size=size),
            pilTransforms.Normalize(scaleIn=1)
        ]
    )
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.PILDataset(data['trainIm'], data['trainAnnot'], Double=False, transform=train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.PILDataset(data['valIm'], data['valAnnot'], Double=False, transform=val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=num_work, pin_memory=True)

    return trainLoader, valLoader, data


def cityPIL_randcrop( cached_data_file, data_dir, classes, batch_size, num_work=6,
                size=(1024, 512), crop_size=(1024, 512),  scale1 = 1, ignore_idx=255):
    print("This input size is  " +str(size))

    if not os.path.isfile(cached_data_file):
        dataLoad = ld.LoadData(data_dir, classes, cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(cached_data_file, "rb"))


    train_transforms = pilTransforms.Compose(
        [
            pilTransforms.Resize(size=size),
            pilTransforms.RandomCrop(crop_size=crop_size,ignore_idx=ignore_idx),
            pilTransforms.RandomFlip(),
            pilTransforms.Normalize(scaleIn=scale1)
        ]
    )
    val_transforms = pilTransforms.Compose(
        [
            pilTransforms.Resize(size=(2048,1024)),
            pilTransforms.Normalize(scaleIn=1)
        ]
    )
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.PILDataset(data['trainIm'], data['trainAnnot'], Double=False, transform=train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.PILDataset(data['valIm'], data['valAnnot'], Double=False, transform=val_transforms),
        batch_size=batch_size//2, shuffle=False, num_workers=num_work, pin_memory=True)

    return trainLoader, valLoader, data

def cityCV_dataloader(cached_data_file, data_dir, classes, batch_size, scaleIn, size=1024, num_work=6):
    if size == 1024:
        scale = [1024, 1536, 1280, 768, 512]
        crop = [32,96,96,32,12]
    elif size == 2048:
        scale = [2048, 1536, 1280, 1024, 768]
        crop = [96,96,64,32,32]

    else:
        scale = [1024, 1536, 1280, 768, 512]
        crop = [32,100,100,32,0]


    if not os.path.isfile(cached_data_file):
        dataLoad = ld.LoadData(data_dir, classes, cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(cached_data_file, "rb"))

    trainDataset_main = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[0],scale[0]//2), #(1024, 512),
        cvTransforms.RandomCropResize(crop[0]), #(32),
        cvTransforms.RandomFlip(),
        # cvTransforms.RandomCrop(64).
        cvTransforms.ToTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[0],scale[0]//2,crop[0]))

    trainDataset_scale1 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[1],scale[1]//2),  # 1536, 768
        cvTransforms.RandomCropResize(crop[1]),
        cvTransforms.RandomFlip(),
        # cvTransforms.RandomCrop(64),
        cvTransforms.ToTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[1],scale[1]//2,crop[1]))

    trainDataset_scale2 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[2],scale[2]//2),  # 1536, 768
        cvTransforms.RandomCropResize(crop[2]),
        cvTransforms.RandomFlip(),
        # cvTransforms.RandomCrop(64),
        cvTransforms.ToTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[2],scale[2]//2,crop[2]))

    trainDataset_scale3 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[3],scale[3]//2), #(768, 384),
        cvTransforms.RandomCropResize(crop[3]),
        cvTransforms.RandomFlip(),
        # cvTransforms.RandomCrop(64),
        cvTransforms.ToTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[3],scale[3]//2,crop[3]))

    trainDataset_scale4 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[4],scale[4]//2),#(512, 256),
        cvTransforms.RandomCropResize(crop[4]),
        cvTransforms.RandomFlip(),
        # cvTransforms.RandomCrop(64).
        cvTransforms.ToTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[4],scale[4]//2,crop[4]))

    valDataset = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[0],scale[0]//2), #(1024, 512),
        cvTransforms.ToTensor(scaleIn),
        #
    ])
    print("%d , %d image size validation" %(scale[0],scale[0]//2))


    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale1),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    trainLoader_scale2 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale2),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    trainLoader_scale3 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale3),
        batch_size=batch_size + 4, shuffle=True, num_workers=num_work, pin_memory=True)

    trainLoader_scale4 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale4),
        batch_size=batch_size + 4, shuffle=True, num_workers=num_work, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=batch_size, shuffle=False, num_workers=num_work, pin_memory=True)

    return trainLoader, trainLoader_scale1, trainLoader_scale2, trainLoader_scale3, trainLoader_scale4, valLoader, data


def cityCVaux_dataloader(cached_data_file, data_dir, classes, batch_size, scaleIn, size=1024, num_work=6):
    if size == 1024:
        scale = [1024, 1536, 1280, 768, 512]
        crop = [32,96,96,32,12]
    elif size == 2048:
        scale = [2048, 1536, 1280, 1024, 768]
        crop = [96,96,64,32,32]

    else:
        scale = [1024, 1536, 1280, 768, 512]
        crop = [32,100,100,32,0]


    if not os.path.isfile(cached_data_file):
        dataLoad = ld.LoadData(data_dir, classes, cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(cached_data_file, "rb"))

    trainDataset_main = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[0],scale[0]//2), #(1024, 512),
        cvTransforms.RandomCropResize(crop[0]), #(32),
        cvTransforms.RandomFlip(),
        cvTransforms.ToMultiTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[0],scale[0]//2,crop[0]))

    trainDataset_scale1 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[1],scale[1]//2),  # 1536, 768
        cvTransforms.RandomCropResize(crop[1]),
        cvTransforms.RandomFlip(),
        cvTransforms.ToMultiTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[1],scale[1]//2,crop[1]))

    trainDataset_scale2 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[2],scale[2]//2),  # 1536, 768
        cvTransforms.RandomCropResize(crop[2]),
        cvTransforms.RandomFlip(),
        cvTransforms.ToMultiTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[2],scale[2]//2,crop[2]))

    trainDataset_scale3 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[3],scale[3]//2), #(768, 384),
        cvTransforms.RandomCropResize(crop[3]),
        cvTransforms.RandomFlip(),
        cvTransforms.ToMultiTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[3],scale[3]//2,crop[3]))

    trainDataset_scale4 = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[4],scale[4]//2),#(512, 256),
        cvTransforms.RandomCropResize(crop[4]),
        cvTransforms.RandomFlip(),
        cvTransforms.ToMultiTensor(scaleIn),
        #
    ])
    print("%d , %d image size train with %d crop" %(scale[4],scale[4]//2,crop[4]))

    valDataset = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(scale[0],scale[0]//2), #(1024, 512),
        cvTransforms.ToMultiTensor(1),
        #
    ])
    print("%d , %d image size validation" %(scale[0],scale[0]//2))


    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyAuxDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        myDataLoader.MyAuxDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale1),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    trainLoader_scale2 = torch.utils.data.DataLoader(
        myDataLoader.MyAuxDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale2),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    trainLoader_scale3 = torch.utils.data.DataLoader(
        myDataLoader.MyAuxDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale3),
        batch_size=batch_size + 4, shuffle=True, num_workers=num_work, pin_memory=True)

    trainLoader_scale4 = torch.utils.data.DataLoader(
        myDataLoader.MyAuxDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale4),
        batch_size=batch_size + 4, shuffle=True, num_workers=num_work, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyAuxDataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=batch_size-2, shuffle=False, num_workers=num_work, pin_memory=True)

    return trainLoader, trainLoader_scale1, trainLoader_scale2, trainLoader_scale3, trainLoader_scale4, valLoader, data


def cityCV_randscalecrop(cached_data_file, data_dir, classes, batch_size, size_h, size_w, scale, num_work=6):
    print("This input size is  %d , %d" %(size_h, size_w))
    if not os.path.isfile(cached_data_file):
        dataLoad = ld.LoadData(data_dir, classes, cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(cached_data_file, "rb"))

    trainDataset_main = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(size_w, size_h),
        cvTransforms.RandomCropResize(32),
        cvTransforms.RandomFlip(),
        # cvTransforms.RandomCrop(64).
        cvTransforms.ToTensor(scale),
        #
    ])


    valDataset = cvTransforms.Compose([
        cvTransforms.Normalize(mean=data['mean'], std=data['std']),
        cvTransforms.Scale(size_w, size_h),
        cvTransforms.ToTensor(1),
        #
    ])


    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main),
        batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=batch_size, shuffle=False, num_workers=num_work, pin_memory=True)

    return trainLoader, valLoader, data


# def get_dataloader(dataset_name, cached_data_file, data_dir, classes, batch_size, scaleIn=1, size_h=512, aux=False):
def get_dataloader(args):
    dataset_name = args["dataset_name"]
    data_file= args["cached_data_file"]
    cls = args["classes"]
    bs = args["batch_size"]
    size_w = args["baseSize"]
    size_h = size_w//2

    print(" Train %s loader, call %s data_file , cls = %d" %(dataset_name, data_file, cls))

    if dataset_name =='citymultiscaleCV':
        num_work= args["num_work"]

        return cityCV_dataloader(data_file, args["data_dir"], cls, bs, args["scaleIn"],size_w,num_work)

    elif dataset_name =='cityCVaux_dataloader':
        num_work= args["num_work"]

        return cityCVaux_dataloader(data_file, args["data_dir"], cls, bs, args["scaleIn"],size_w,num_work)


    elif dataset_name =='cityCV':
        num_work = args["num_work"]
        scale1 = args["scaleIn"]

        return cityCV_randscalecrop(data_file, args["data_dir"], cls, bs, size_h, size_w, scale1, num_work)

    elif dataset_name =="citypilAux":
        scale1 = args["scale1"]
        scale2 = args["scale2"]
        num_work= args["num_work"]

        return cityPIL_Doublerandscalecrop(data_file, args["data_dir"], cls, bs, num_work=num_work,
                 scale=(0.5, 2.0), size=(size_w, size_h), scale1 = scale1, scale2= scale2, ignore_idx=19)

    elif dataset_name =="citypil":
        scale1 = args["scaleIn"]
        num_work= args["num_work"]

        return cityPIL_randscalecrop(data_file, args["data_dir"], cls, bs, num_work=num_work,
                 scale=(0.5, 2.0), size=(size_w, size_h), scale1 = scale1, ignore_idx=19)

    elif dataset_name == "citypilcrop":
        scale1 = args["scaleIn"]
        num_work = args["num_work"]
        crop_size = args["crop_size"]

        return cityPIL_randcrop(data_file, args["data_dir"], cls, bs, num_work=num_work,
                                   size=(size_w, size_w), crop_size=(crop_size, crop_size), scale1=scale1, ignore_idx=19)

    else:
        print(dataset_name + "is not implemented")
        raise NotImplementedError
