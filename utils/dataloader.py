import os
import json
import numpy as np

from argparse import ArgumentParser as Parser
from torch.utils.data import DataLoader, Dataset

from . import transforms as T


class MyDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_list: list,
                 transforms: object):
        ''' Args:
        * `data_dir`: save path of dataset.
        * `data_list`: filenames of data in this fold.
        * `transforms`: transform functions for dataset.
        '''
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.data_list = data_list
        self.transforms = transforms

    def __getitem__(self, index: int):
        ''' get image, label and filename by index '''
        info = {"image": "", "label": None}
        for key, val in self.data_list[index].items():
            info[key] = os.path.join(self.data_dir, val)
        img, msk = self.transforms(info["image"], info["label"])
        info["fname"] = os.path.basename(info["image"])
        return img, msk, info

    def __len__(self):
        ''' get length of the dataset '''
        return len(self.data_list)


def get_transforms(args: Parser, is_test: bool = False):
    if not is_test:
        tra_transfoms = T.Compose([
            T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
            T.RandomCrop(crop_size=(args.roi_z, args.roi_y, args.roi_x)),
            T.ScaleIntensity(scope=(-1200, 600), range=(-1, 1), clip=True),
            T.RandomFilp(prob=0.2, axes=(0, 1, 2)),
            T.RandomRot90(prob=0.2, axes=(0, 1, 2)),
            T.RandomScaleIntensity(prob=0.1, factor=0.1),
            T.RandomShiftIntensity(prob=0.1, offset=0.1),
            T.AddChannel(img_add=True, msk_add=True),
            T.ToTensor()
        ])
        val_transforms = T.Compose([
            T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
            T.ScaleIntensity(scope=(-1200, 600), range=(-1, 1), clip=True),
            T.AddChannel(img_add=True, msk_add=True),
            T.ToTensor()
        ])
        return (tra_transfoms, val_transforms)
    else:
        test_transforms = T.Compose([
            T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
            T.ScaleIntensity(scope=(-1200, 600), range=(-1, 1), clip=True),
            T.AddChannel(img_add=True, msk_add=True),
            T.ToTensor()
        ])
        return test_transforms


def get_loader(args: Parser, is_test: bool = False):
    transforms = get_transforms(args, is_test)

    if not is_test:
        data_list = get_data_list(args, "training")
        tra_data, val_data = split_data_list(
            data_list, args.num_folds, fold=args.fold
        )
        train_dataset = MyDataset(data_dir=args.data_root,
                                  data_list=tra_data,
                                  transforms=transforms[0])
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

        val_dataset = MyDataset(data_dir=args.data_root,
                                data_list=val_data,
                                transforms=transforms[1])
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers)
        return train_loader, val_loader
    else:
        test_data = get_data_list(args, "test")
        # if no test data, then using val data
        if len(test_data) == 0:
            _, test_data = split_data_list(
                get_data_list(args, "training"), 
                args.num_folds, fold=args.fold
            )
        test_dataset = MyDataset(data_dir=args.data_root,
                                 data_list=test_data,
                                 transforms=transforms)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=args.num_workers)
        return test_loader


def get_data_list(args: Parser, key: str):
    json_path = os.path.join(args.data_root, args.json_name)
    with open(json_path, "r") as jsf:
        json_data = json.load(jsf)[key]
    assert isinstance(json_data, list) and len(json_data) > 0

    if not isinstance(json_data[0], dict):
        for idx, val in enumerate(json_data):
            json_data[idx] = {"image": val}
    else:   # check format of `json_data`
        for idx, val in enumerate(json_data):
            assert "image" in json_data[idx].keys()
            assert "label" in json_data[idx].keys()
    return json_data


def split_data_list(data_list: list, num_folds: int, fold: int = 0):
    assert fold < num_folds, "`fold` should be less than `num_folds`."
    tra_data, val_data = [], []
    for idx, val in enumerate(data_list):
        if idx % num_folds == fold:
            val_data.append(val)
        else:
            tra_data.append(val)
    return tra_data, val_data
