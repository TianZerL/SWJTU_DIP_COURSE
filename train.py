from torch.utils.data import DataLoader
from dataLoader import ImageData
from module import *
from utils.image.processor import *
from utils import calculate
import numpy as np
import torch as tc
import os
from visdom import Visdom

trainDataFolder = r".\data\train128"
testDataFolder = r".\data\test"
moduleFolder = r".\module"
currModuleName = "ACNet"

criteria = nn.L1Loss()
learnRate = 1e-3
weightDecay = 1e-5


if not os.path.exists(moduleFolder):
    os.makedirs(moduleFolder)


def getModulePath(moduleName):
    return moduleFolder + "/" + moduleNames[moduleName]


processorList = [Downscale(), AddNoise(noiseLevel=1)]
#processorList = [Downscale()]

trainDataSet = ImageData(
    trainDataFolder,
    crop=True,
    cropSize=64,
    processorList=processorList,
    colorFromat=moduleColorFormat[currModuleName],
)
testDataSet = ImageData(
    testDataFolder,
    crop=False,
    cropSize=96,
    processorList=processorList,
    colorFromat=moduleColorFormat[currModuleName],
)
trainData = DataLoader(trainDataSet, batch_size=16, shuffle=True)
testData = DataLoader(testDataSet, batch_size=16, shuffle=True)

module = modules[currModuleName]
module = module.cuda()
if os.path.exists(getModulePath(currModuleName)):
    module.load_state_dict(tc.load(getModulePath(currModuleName)))

optimizer = tc.optim.Adam(
    module.parameters(), lr=learnRate, weight_decay=weightDecay, amsgrad=True
)

scheduler = tc.optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer, 
    milestones=[30, 60, 90, 120, 150, 180, 210, 230, 260, 290, 320, 350, 380, 410, 440, 470, 500], 
    gamma=0.9,
)


# 将窗口类实例化
viz = Visdom() 

# 创建窗口并初始化
viz.line([0.], [0], win='train_loss', opts=dict(title=currModuleName+' train loss', legend=['loss']))
viz.line([0.], [0], win='train_pnsr', opts=dict(title=currModuleName+' train pnsr', legend=['pnsr']))
viz.line([0.], [0], win='test_loss', opts=dict(title=currModuleName+' test loss', legend=['loss']))
viz.line([0.], [0], win='test_pnsr', opts=dict(title=currModuleName+' test pnsr', legend=['pnsr']))

# Train
totalEpoch = 500
count = 0
for i in range(totalEpoch):
    module.train()
    trainLoss = []
    trainPsnr = []
    for (lr, hr) in trainData:
        lr, hr = lr.cuda(), hr.cuda()
        optimizer.zero_grad()
        out = module(lr)#.clamp(0.0, 1.0)
        loss = criteria(out, hr)
        loss.backward()
        optimizer.step()
        scheduler.step()

        trainLoss.append(loss.item())
        trainPsnr.append(calculate.psnr(out, hr).item())

        count += 1
        if (count + 1) % 50 == 0:
            tc.save(module.state_dict(), getModulePath(currModuleName))
            #tc.save(trainLoss, "trainLoss.pt")
            #tc.save(trainPsnr, "trainPsnr.pt")

    viz.line([np.mean(trainLoss)], [i], win='train_loss', update='append')
    viz.line([np.mean(trainPsnr)], [i], win='train_pnsr', update='append')

    module.eval()
    with tc.no_grad():
        testLoss = []
        testPsnr = []
        for (lr, hr) in testData:
            lr, hr = lr.cuda(), hr.cuda()
            out = module(lr)#.clamp(0.0, 1.0)
            lossTest = criteria(out, hr)

            testLoss.append(lossTest.item())
            testPsnr.append(calculate.psnr(out, hr).item())

        #tc.save(testLoss, "testLoss.pt")
        #tc.save(testPsnr, "testPsnr.pt")

    viz.line([np.mean(testLoss)], [i], win='test_loss', update='append')
    viz.line([np.mean(testPsnr)], [i], win='test_pnsr', update='append')

    print(
        "epoch {}\ntrain: loss {}, psnr {}\ntest: loss {}, psnr {}".format(
            i, np.mean(trainLoss), np.mean(trainPsnr), np.mean(testLoss), np.mean(testPsnr)
        )
    )
