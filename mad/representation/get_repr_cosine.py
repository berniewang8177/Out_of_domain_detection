import pandas as pd
import numpy as np

import torch

from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from mad.representation.networks import EmbeddingNetLarge, EmbeddingNetMiddle, EmbeddingNetSmall, SiameseNet
from mad.representation.trainer import fit
from mad.representation.losses import EasyPositiveSemiHardNegativeLossCosine, EasyPositiveHardNegativeLossCosine, SCTLossCosine
from mad.representation.losses import Supervised_NT_xent
from mad.representation.datasets import BatchPairSampling

cuda = torch.cuda.is_available()

def train_representation_cosine(data, label, dataset_name, loss = 'cosine'):


    # loss_fn = EasyPositiveSemiHardNegativeLossCosine( )

    # loss_fn = EasyPositiveHardNegativeLossCosine()
    # loss_fn =  SCTLossCosine( method = 'sct', lam = 1)

    # loss_fn = EasyPositiveHardNegativeLossMaha()

    temperature = 0.11


    if dataset_name == 'fried':
        loss_fn = Supervised_NT_xent(temperature)
        # for xent  1, 50p, 15 temp =0.11
        lr = 1*1e-3

        n_epochs = 50
        hidden_size = 10
        embedding_size = 5
        input_size = data.shape[1]
        interval = 10 # useless
        batch_size = 15#8
        net = EmbeddingNetMiddle(input_size, hidden_size, embedding_size)

    if dataset_name == 'diffu':
        # 0.5 with 16 is pretty good
        loss_fn = Supervised_NT_xent(0.5)
        lr =  0.3*1e-3 # 0.7*1e-3 # 0.5

        n_epochs = 40 # 35 # 25
        hidden_size = 100
        embedding_size = 20
        input_size = data.shape[1]
        interval = 10 # useless
        batch_size = 16 # 12 is good 16 is the best
        net = EmbeddingNetSmall(input_size, hidden_size, embedding_size)

    if dataset_name == 'supercond':
        loss_fn = Supervised_NT_xent(0.5)
        # best param: 1 lr, 60ep, 50 batch for cosine CTR
        # best param for xent 1 lr, 60ep, 35
        lr = 1*1e-3# 1*1e-4 # 0.5*1e-3

        n_epochs = 60
        hidden_size = 100
        embedding_size = 20
        input_size = data.shape[1]
        interval = 10 # useless
        batch_size = 35 #60 # 45 # 35 25 #
        net = EmbeddingNetLarge(input_size, hidden_size, embedding_size)

    siamese_net = SiameseNet(  embedding_net = net )
    model = siamese_net
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    datasampler = BatchPairSampling( dataset = data , labels= label)
    loader = torch.utils.data.DataLoader(datasampler, batch_size=batch_size, shuffle=True)

    fit(loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, interval)

    return model
