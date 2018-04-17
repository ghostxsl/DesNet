import torch
from PIL import Image
from torch.autograd import Variable
import numpy as np
import time

from transformer import TransformerNet

net = TransformerNet()
use_cuda = torch.cuda.is_available()
gpu = 0
if use_cuda:
    net = net.cuda(gpu)

net.load_state_dict(torch.load('Gepoch_10.model'))
input_image = Image.open("data/SAR/tra/train_VV_100.tiff")
x = torch.from_numpy(np.array(input_image, np.float32, copy=False))
x = x.view(input_image.size[0], input_image.size[1], 1)
x = x.transpose(0, 1).transpose(0, 2).contiguous()
x = x.view(1, 1, input_image.size[0], input_image.size[1])
net.eval()
if use_cuda:
    x = x.cuda(gpu)
images = Variable(x, volatile=True)

time_start=time.time()
y = net(images)
time_end=time.time()
t=time_end-time_start
print(t)
y = y.view(input_image.size[0], input_image.size[1])
scallop = y.cpu().data.numpy() + 1
x = x.view(input_image.size[0], input_image.size[1]).cpu().numpy()
descallop = x / (scallop + 1e-12)

x = Image.fromarray(x)
x.show()

descallop = Image.fromarray(descallop)
descallop.show()

scallop = scallop / scallop.max() * 255
scallop = Image.fromarray(scallop)
scallop.show()
