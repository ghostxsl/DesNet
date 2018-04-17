import time
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from transformer import TransformerNet

BATCH_SIZE = 8  # Batch size
EPOCHS = 10
L2_Weight = 0.95
Pattern_Weight = 0.05

netG = TransformerNet()
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
    netG = netG.cuda(gpu)

if use_cuda:
    kwargs = {'num_workers': 0, 'pin_memory': False}
else:
    kwargs = {}

def train():
    train_dataset = datasets.SarImage("data/SAR/train_VV.txt")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-2)
    mse_loss = torch.nn.MSELoss().cuda(gpu)

    #training data
    for e in range(EPOCHS):
        netG.train()
        count = 0
        for batch_id, (x, lab) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            netG.zero_grad()
            if use_cuda:
                x = x.cuda(gpu)
                lab = lab.cuda(gpu)
            x = Variable(x)
            lab = Variable(lab)
            fake = netG(x)
            L2_loss = L2_Weight * mse_loss(fake, lab)
            Pattern_loss = Pattern_Weight * mse_loss(fake.mean(3), lab.mean(3))
            Total_loss = L2_loss + Pattern_loss
            Total_loss.backward()
            optimizerG.step()
            mesg = "{}\tEpoch {}:\t[{}/{}]\tloss: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset), L2_loss.data[0] + Pattern_loss.data[0]
                    )
            print(mesg)
            print()
    # save model
    netG.cpu()
    save_model_filename = "Gepoch_" + str(EPOCHS) + ".model"
    torch.save(netG.state_dict(), save_model_filename)

    print("\nDone, trained model saved at", save_model_filename)

if __name__ == "__main__":
    train()
