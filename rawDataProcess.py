from utils.image.reduce import *
from PIL import Image
import os, uuid

rawPath = r".\data\raw"
reducedPath = r".\data\128"
times = 1

processor = RawDataProcessor3(cropSize=128)
imgs = getAllImages(rawPath)

if not os.path.exists(reducedPath):
    os.makedirs(reducedPath)

def process():
    for i in imgs:
        img = Image.open(i)
        img = processor.process(img)
        tmpName = uuid.uuid4().hex
        img.save(reducedPath + "/" + tmpName[:6] + ".png", format="PNG", quality=100)

for i in range(times):
    process()
