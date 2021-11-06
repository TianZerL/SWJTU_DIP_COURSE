from torchvision.transforms import RandomCrop
from PIL import Image
from io import BytesIO
import random

colorFormats = ["RGB", "YUV", "L"]

class ImagePreprocessor(object):
    def __init__(self, crop=True, cropSize=96, colorFromat="RGB", processorList=None):
        self.cropper = RandomCrop(cropSize) if crop else None
        self.colorFormat = colorFromat
        self.processorList = processorList

    def preReduce(self, img, cvtTo="RGB"):
        if self.cropper is not None:
            img = self.cropper(img)
        if self.processorList is None:
            raise IndexError("processorList can't be None")
        for processor in self.processorList:
            img = processor(img)
        if cvtTo == "RGB":
            img = img.convert("RGB")
        elif cvtTo == "YCbCr":
            img = img.convert("YCbCr")
        elif cvtTo == "L":
            img = img.convert("L")
        return img

    def preReduce2(self, img, cvtTo="RGB"):
        if self.processorList is None:
            raise IndexError("processorList can't be None")
        for processor in self.processorList:
            img = processor(img)
        if self.cropper is not None:
            img = self.cropper(img)
        if cvtTo == "RGB":
            img = img.convert("RGB")
        elif cvtTo == "YCbCr":
            img = img.convert("YCbCr")
        elif cvtTo == "L":
            img = img.convert("L")
        return img

    def process(self, img):
        if self.cropper is not None:
            img = self.cropper(img)
        hr = img.copy()
        lr = img
        if self.processorList is None:
            self.processorList = [Downscale(), AddNoise()]
        for processor in self.processorList:
            lr = processor(lr)
        if self.colorFormat == "RGB":
            lr = lr.convert("RGB")
            hr = hr.convert("RGB")
        elif self.colorFormat == "L":
            lr = lr.convert("L")
            hr = hr.convert("L")
        elif self.colorFormat == "YUV":
            lr, _, _ = lr.convert("YCbCr").split()
            hr = hr.convert("YCbCr")
        return lr, hr

def reMergeYUV(lr, hr):
    _, U, V = hr.convert("YCbCr").split()
    Y = lr
    lr = Image.merge("YCbCr", (Y, U, V))
    return lr


class Downscale(object):
    def __init__(self, factor=2, flag=None):
        self.factor = factor
        if flag is None:
            flag = random.choice([Image.BICUBIC, Image.LANCZOS, Image.BILINEAR])
        self.flag = flag

    def __call__(self, img):
        w, h = tuple(map(lambda x: x // self.factor, img.size))
        img = img.resize((w, h), self.flag)
        return img


class Upscale(object):
    def __init__(self, factor=2, flag=None):
        self.factor = factor
        if flag is None:
            flag = random.choice([Image.BICUBIC, Image.LANCZOS, Image.BILINEAR])
        self.flag = flag

    def __call__(self, img):
        w, h = tuple(map(lambda x: x * self.factor, img.size))
        img = img.resize((w, h), self.flag)
        return img


class AddNoise(object):
    def __init__(self, noiseLevel=1):
        if noiseLevel == 1:
            self.noiseLevel = [5, 25]
        elif noiseLevel == 2:
            self.noiseLevel = [25, 50]
        elif noiseLevel == 3:
            self.noiseLevel = [50, 75]
        elif noiseLevel == 4:
            self.noiseLevel = [75, 95]
        else:
            raise KeyError("Noise level should in 1, 2")

    def __call__(self, img):
        quality = 100 - round(random.uniform(*self.noiseLevel))
        tmpIO = BytesIO()
        img.save(tmpIO, format="JPEG", quality=quality)
        tmpIO.seek(0)
        img = Image.open(tmpIO)
        return img
