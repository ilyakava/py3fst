# -*- coding: utf-8 -*-

import torch
import pdb


#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
H, W, D = 145, 145, 200

# batch size, num input channesls
x = torch.autograd.Variable(torch.randn(1, 1, H, W, D).type(dtype))
g = torch.ones(9,9,5)

model = torch.nn.Sequential(
    torch.nn.Conv3d(1, 75, (9,9,5), (1, 1, 5), (4,4,2)),
    torch.nn.ReLU()
)

#model.weight.data.fill_(1)
#model.bias.data.fill_(0)
# model.parameters()

model.cuda()
model.eval()

output = model(x)

out2 = model(output)
pdb.set_trace()

