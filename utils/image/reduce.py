from utils.image.processor import *
import glob


class RawDataProcessor(object):
    def __init__(self, downscaleFactor=2, flag=Image.LANCZOS, cropSize=256):
        processorList = [Downscale(factor=downscaleFactor, flag=flag)]
        self.preprocessor = ImagePreprocessor(crop=False, processorList=processorList)
        self.processor = ImagePreprocessor(
            crop=True, cropSize=cropSize, processorList=processorList
        )

    def process(self, image):
        image = self.preprocessor.preReduce(image)
        image = self.processor.preReduce(image)
        return image

class RawDataProcessor2(object):
    def __init__(self, cropSize=256):
        processorList = []
        self.processor = ImagePreprocessor(
            crop=True, cropSize=cropSize, processorList=processorList
        )

    def process(self, image):
        image = self.processor.preReduce(image)
        return image

class RawDataProcessor3(object):
    def __init__(self, cropSize=128):
        processorList = [Downscale(factor=4, flag=Image.LANCZOS)]
        self.processor = ImagePreprocessor(
            crop=True, cropSize=cropSize, processorList=processorList
        )

    def process(self, image):
        image = self.processor.preReduce2(image)
        return image

def getAllImages(path):
    imgs = glob.glob(path + "/**", recursive=True)
    imgs = filter(
        lambda path: path.endswith("png")
        or path.endswith("jpg")
        or path.endswith("jpeg"),
        imgs,
    )
    return list(imgs)
