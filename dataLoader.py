from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from utils.image.processor import ImagePreprocessor, colorFormats
from PIL import Image
import glob, random


class ImageData(Dataset):
    def __init__(
        self, srcPath, crop=True, cropSize=96, colorFromat="RGB", processorList=None
    ):
        super(ImageData, self).__init__()
        imgs = glob.glob(srcPath + "/**", recursive=True)
        imgs = filter(
            lambda path: path.endswith("png")
            or path.endswith("jpg")
            or path.endswith("jpeg"),
            imgs,
        )
        self.imgs = list(imgs)
        self.imgNum = len(self.imgs)
        if colorFromat not in colorFormats:
            raise KeyError("only RGB or YUV or L")
        self.imgPreprocessor = ImagePreprocessor(
            crop=crop,
            cropSize=cropSize,
            colorFromat=colorFromat,
            processorList=processorList,
        )

    def __len__(self):
        return self.imgNum

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        img = Image.open(imgPath)
        lr, hr = self.imgPreprocessor.process(img)
        return to_tensor(lr), to_tensor(hr)
