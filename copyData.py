from shutil import copy
import os
import glob

reducePath = r".\data\reduced"
trainDataFolder = r".\data\train"
testDataFolder = r".\data\test"

if not os.path.exists(trainDataFolder):
    os.makedirs(trainDataFolder)
if not os.path.exists(testDataFolder):
    os.makedirs(testDataFolder)

imgs = glob.glob(reducePath+"/**", recursive=True)
imgNum = len(imgs)
print(imgNum)
trainDataNum = imgNum * 0.85
testDataNum = imgNum - trainDataNum
print(trainDataNum)
print(testDataNum)
count = 1

while count < trainDataNum:
    copy(imgs[count], trainDataFolder)
    count+=1

while count < imgNum:
    copy(imgs[count], testDataFolder)
    count+=1
