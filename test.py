from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from module import *
import torch as tc
import time

path = r"F:\Temp\Anime4K\p1.png"
currModuleName = "ACNet"
#currModuleName = "Waifu2x"
moduleFolder = r".\module"
Lflag = True

org = Image.open(path)
img = org.convert(moduleColorFormat[currModuleName])
img = to_tensor(img).unsqueeze(0)


def getModulePath(moduleName):
    return moduleFolder + "/" + moduleNames[moduleName]

module = modules[currModuleName]
module = module.cuda()
module.load_state_dict(tc.load(getModulePath(currModuleName)))

module.eval()
with tc.no_grad():
    img = img.cuda()
    t = time.time()
    img = module(img)
    print(time.time()-t)
    img = img.squeeze(0)
    img = img.cpu().clamp(0.0, 1.0)
    img = to_pil_image(img)
    if moduleColorFormat[currModuleName] == "L" and Lflag:
        _, U, V = org.convert("YCbCr").split()
        U = U.resize((U.size[0]*2, U.size[1]*2), Image.LANCZOS)
        V = V.resize((V.size[0]*2, V.size[1]*2), Image.LANCZOS)
        img = Image.merge("YCbCr", (img, U, V))
    img.show()
    #img.save(r"F:\Temp\Anime4K\test.jpg",quality = 100)
