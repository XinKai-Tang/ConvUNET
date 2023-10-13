import os
import numpy as np
import SimpleITK as sitk

from random import randint
from typing import Sequence, Tuple
from scipy import ndimage as ndi

from skimage.filters import roberts
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_erosion, disk


def crop_image(img: np.ndarray, new_size: Sequence,
               rand_crop: bool = False, msk: np.ndarray = None):
    ''' crop the image and mask '''
    old_size = img.shape
    if any(o < n for o, n in zip(old_size, new_size)):
        raise ValueError("new_size should not be larger than the image size.")
    # 获取图像裁剪的范围：
    if rand_crop:   # 随机裁剪
        low = [randint(0, int(o-n)) for o, n in zip(old_size, new_size)]
        high = [int(l+n) for l, n in zip(low, new_size)]
    else:           # 中心裁剪
        low = [int((o-n)/2) for o, n in zip(old_size, new_size)]
        high = [int(l+n) for l, n in zip(low, new_size)]
    # 裁剪图像和掩膜：
    if msk is None:
        if len(old_size) == 2:
            new_img = img[low[0]:high[0], low[1]:high[1]]
        else:
            new_img = img[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
        return new_img
    else:
        if len(old_size) == 2:
            new_img = img[low[0]:high[0], low[1]:high[1]]
            new_msk = msk[low[0]:high[0], low[1]:high[1]]
        else:
            new_img = img[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
            new_msk = msk[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
        return new_img, new_msk


def change_orientation(src: sitk.Image, orient: str = 'RAS'):
    ''' change orientation of 3d image '''
    assert isinstance(orient, str) and len(orient) == 3
    opts = {
        'L': [-1., 0., 0.], 'R': [1., 0., 0.],
        'P': [0., -1., 0.], 'A': [0., 1., 0.],
        'S': [0., 0., -1.], 'I': [0., 0., 1.],
    }
    direction = []
    for o in orient: direction.extend(opts[o])
    filter = sitk.ResampleImageFilter()
    filter.SetSize(src.GetSize())                       # 设置处理后的图像大小
    filter.SetOutputSpacing(src.GetSpacing())           # 设置X,Y,Z方向的间隔
    filter.SetOutputOrigin(src.GetOrigin())             # 设置原点的坐标
    filter.SetOutputDirection(direction)                # 设置坐标系方向
    filter.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    filter.SetInterpolator(sitk.sitkLinear)             # 设置图像的插值方式
    filter.SetOutputPixelType(sitk.sitkInt16)           # 设置数组元素类型
    dst: sitk.Image = filter.Execute(src)
    return dst


def load_as_array(path: str, dtype = np.float32):
    ''' load image from file '''
    if path[-4:] == ".npy":
        arr = np.load(path)
    elif os.path.isdir(path):
        img = load_CT_from_dir(path)
        arr = sitk.GetArrayFromImage(img)
    else:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
    return np.array(arr, dtype=dtype)


def load_CT_from_dir(ct_dir: str):
    ''' 读入路径下的CT并返回精度最高的CT '''
    # (1)分析该文件夹下的所有CT：
    reader = sitk.ImageSeriesReader()           # 构建序列读取器
    reader.MetaDataDictionaryArrayUpdateOn()    # 允许读入公开的元数据
    seriesIDs = list(reader.GetGDCMSeriesIDs(ct_dir))
    # (2)读入切片数量最多的CT：
    seriesIDs.sort(key=lambda sid: 
                   len(reader.GetGDCMSeriesFileNames(ct_dir, sid)))
    dicom_names = reader.GetGDCMSeriesFileNames(ct_dir, seriesIDs[-1])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    # (3)读取Dicom中保存的数据：
    info = {
        "seriesUID": reader.GetMetaData(0, '0020|000e'),
        "patientID": reader.GetMetaData(0, '0010|0020'),
        "studyDate": reader.GetMetaData(0, '0008|0020')[:8],
        "studyTime": reader.GetMetaData(0, '0008|0030')[:6],
    }
    return image, info


def save_as_file(path: str, arr: np.ndarray, 
                 dtype = np.int16, **args):
    ''' save image as a file '''
    arr = np.array(arr, dtype=dtype)
    if path[-4:] == ".npy":
        np.save(path, arr)
    else:
        img = sitk.GetImageFromArray(arr)
        if args.get("spacing") is not None:
            img.SetSpacing(args["spacing"])
        if args.get("direction") is not None:
            img.SetDirection(args["direction"])
        if args.get("origin") is not None:
            img.SetOrigin(args["origin"])
        sitk.WriteImage(img, path)


def normalize_ct(src: np.ndarray,
                 scope: Tuple = (-1200, 600),
                 range: Tuple = (-1.0, 1.0),
                 clip: bool = True):
    ''' normalize ct values. '''
    assert scope[0] < scope[1], "scope[0] should be smaller than scope[1]."
    assert range[0] < range[1], "range[0] should be smaller than range[1]."
    dst = (src - scope[0]) / float(scope[1] - scope[0])
    dst = dst * float(range[1] - range[0]) + range[0]
    dst = dst.clip(range[0], range[1]) if clip else dst
    return dst.astype(np.float32)


def pixel_to_hu(src: np.ndarray,
                slope: float = 1, intercept: float = 0,
                level: int = -300, width: int = 1800):
    ''' convert pixel values into HU values. '''
    # 调整像素值为HU值：
    src = src * float(slope) + float(intercept)
    # 调整窗位和窗宽：
    min_val = level - width / 2
    max_val = level + width / 2
    src = src.clip(min_val, max_val)
    return src


def resample_ct(src_img: sitk.Image,
                new_spacing: list = [1., 1., 1.],
                src_msk: sitk.Image = None):
    ''' resample the whole ct scan '''
    old_size = src_img.GetSize()
    old_spacing = src_img.GetSpacing()
    new_size = [int(s * o / n)
                for s, o, n in zip(old_size, old_spacing, new_spacing)]
    new_spacing = [(s * o / n) 
                   for o, n, s in zip(old_size, new_size, old_spacing)]
    filter = sitk.ResampleImageFilter()
    filter.SetSize(new_size)                            # 设置处理后的图像大小
    filter.SetOutputSpacing(new_spacing)                # 设置X,Y,Z方向的间隔
    filter.SetOutputOrigin(src_img.GetOrigin())         # 设置原点的坐标
    filter.SetOutputDirection(src_img.GetDirection())   # 设置坐标系方向
    filter.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    filter.SetInterpolator(sitk.sitkLinear)             # 设置图像的插值方式
    filter.SetOutputPixelType(sitk.sitkInt16)           # 设置数组元素类型
    dst_img: sitk.Image = filter.Execute(src_img)
    if src_msk is None:
        return dst_img
    else:
        filter.SetInterpolator(sitk.sitkNearestNeighbor)    # 设置掩膜的插值方式
        filter.SetOutputPixelType(sitk.sitkUInt8)           # 设置数组元素类型
        dst_msk: sitk.Image = filter.Execute(src_msk)
        return dst_img, dst_msk


def get_lung_bbox(bin_mask: np.ndarray):
    ''' get bbox of the lung region '''
    z_true, y_true, x_true = np.where(bin_mask)
    lung_box = [[np.min(z_true), np.max(z_true)],
                [np.min(y_true), np.max(y_true)],
                [np.min(x_true), np.max(x_true)]]
    return lung_box


def segment_ct(img3d: np.ndarray, threshold: int = -420, margin: int = 8):
    ''' segment the lungs from the whole ct scan '''
    seg3d = np.asarray([
        segment_slice(s, threshold) for s in img3d
    ], dtype="bool")
    struct = ndi.generate_binary_structure(3, 1)
    seg3d = ndi.binary_dilation(seg3d, struct, margin)
    img3d = np.where(seg3d > 0, img3d.clip(-1400, 700), -1536)
    return img3d


def segment_slice(img2d: np.ndarray, threshold: int = -420):
    ''' segment the lungs from the given 2D slice '''
    # (1) Convert into a binary image.
    img_bin = img2d < threshold
    # (2) Remove the blobs connected to the border of the image.
    img_clr = clear_border(img_bin)
    # (3) Label the image.
    img_lab = label(img_clr)
    # (4) Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(img_lab)]
    areas.sort()
    if len(areas) > 2:      # remove other smaller areas
        for r in regionprops(img_lab):
            if r.area < areas[-2]:
                for coord in r.coords:
                    img_lab[coord[0], coord[1]] = 0
    img_bin = img_lab > 0
    # (5) Seperate nodules attached to the blood vessels.
    img_bin = binary_erosion(img_bin, disk(2))
    # (6) Keep nodules attached to the lung wall.
    img_bin = binary_closing(img_bin, disk(10))
    # (7) Fill in the small holes inside the mask of lungs.
    edges = roberts(img_bin)
    img_bin = ndi.binary_fill_holes(edges)
    # (8) Superimpose the mask on the input image.
    img_bin[img_bin == 0] = False
    img_bin[img_bin == 1] = True
    return img_bin


def world_to_voxel_coord(world_coord: tuple,
                         origin: tuple,
                         spacing: tuple):
    ''' convert world coordination into voxel coordination '''
    stretched_coord = np.abs(world_coord - origin)
    voxel_coord = stretched_coord / spacing
    return voxel_coord
