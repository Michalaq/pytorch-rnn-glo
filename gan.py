import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from functools import reduce
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import math

def sine_dataset(n, l, d):
    inputs = [Variable(torch.from_numpy(np.random.normal(size=(1,d)).astype(np.float32)), requires_grad=True)
              for _ in range(n)]
    targets = [Variable(torch.from_numpy(np.sin(np.linspace(0,2*np.pi,l) * np.random.uniform(1,10)
                + np.random.uniform(-np.pi, np.pi)).astype(np.float32))) for _ in inputs]
    return inputs, targets


class Generator(nn.Module):

    def __init__(self, latent_dim, seq_dim, start_len, factor, target_len, act=nn.ReLU, upsampling=nn.ConvTranspose1d):
        super(Generator, self).__init__()
        assert(target_len % factor == 0 and
               factor > 0 and
               start_len < target_len)
        self.input_dim = start_len
        lays = []
        ups = []
        lay_act = []
        ups_act = []
        self.first_lay = nn.Linear(latent_dim, start_len)
        self.first_act = nn.ReLU()
        tmp_dim = target_len * seq_dim
        lays.append(nn.GRU(1, tmp_dim, batch_first=True))
        cur_len = start_len
        for i in range(int(math.log(target_len // cur_len, factor))):
            lay_act.append(act())
            ups.append(upsampling(tmp_dim, tmp_dim // factor, stride=factor, kernel_size=2*factor, padding=factor//2))
            ups_act.append(act())
            tmp_dim //= factor
            cur_len *= factor
            if cur_len < target_len:
                lays.append(nn.GRU(tmp_dim, tmp_dim, batch_first=True))
        self.final_lay = nn.Conv1d(tmp_dim, seq_dim, 1)
        self.layers = nn.ModuleList(lays)
        self.lay_act = lay_act
        self.ups_act = ups_act
        self.ups = nn.ModuleList(ups)

    def forward(self, input):
        x = input
        x = self.first_lay(x)
        x = self.first_act(x)
        # print(x.size())
        x = x.view(x.size(0), -1, 1)
        for rnn, act1, ups, act2 in zip(self.layers, self.lay_act, self.ups, self.ups_act):
            x, _ = rnn(x)
            # print(x.size())
            x = act1(x)
            x = x.transpose(-1, -2)
            x = ups(x)
            # print(x.size())
            x = x.transpose(-1, -2)
            x = act2(x)
        x = x.transpose(-1, -2)
        x = self.final_lay(x)
        x = x.transpose(-1, -2)
        return x


def to_one_hot(x):
    INPUT_DIM = 10
    tmp = torch.from_numpy(np.zeros(INPUT_DIM).astype(np.float32))
    tmp[x] = 1
    return Variable(tmp.cuda(), requires_grad=True)


def to_random_noise(x):
    INPUT_DIM = 10
    tmp = torch.from_numpy(np.random.uniform(-1,1,size=(INPUT_DIM,)).astype(np.float32))
    return Variable(tmp.cuda(), requires_grad=True)


to_tensor = ToTensor()

def to_variable(x):
    return Variable(to_tensor(x).cuda())


NB_EPOCH = 100
BATCH_SIZE = 128

print('loading mnist dataset...')
ds = MNIST('/home/bartek/pytorch_datasets/', transform=to_variable, target_transform=to_random_noise, download=True)
inputs = [ x for _, x in ds ]
targets = [ x for x, _ in ds ]
print('creating batches...')
batch = lambda data, batch_size: [ torch.stack([data[i] for i in range(j-batch_size, j)])
                                   for j in range(batch_size, len(data), batch_size)]
batched = list(zip(batch(inputs, BATCH_SIZE), batch(targets, BATCH_SIZE)))
print('done.')
print('creating generator...')
g = Generator(10, 28, 7, 2, 28)
g.cuda()
print('generator created: ', g)

crit = nn.MSELoss()
optim = torch.optim.Adam([*inputs, *g.parameters()], lr=0.01)

print('start training...')
for i in range(NB_EPOCH):
    print('epoch',i)
    for j, (x_i, y_i) in enumerate(batched):
        print('batch',j)
        optim.zero_grad()
        out = g(x_i)
        loss = crit(out, y_i)
        loss.backward()
        optim.step()
        print('loss', loss.data[0])
