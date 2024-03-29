import torch.nn as nn

def GlorotOrthogonal(tensor, scale=2.0):
    if tensor is not None:
        nn.init.orthogonal_(tensor.data)
        scale /= (tensor.size(-2) + tensor.size(-1)) * tensor.var()
        tensor.data *= scale.sqrt()

def GlorotOrthogonal_(tensor):
    if tensor is not None:
        nn.init.normal_(tensor.data)


class ResidualLayer(nn.Module):
    def __init__(self, units, activation=None):
        super(ResidualLayer, self).__init__()

        self.activation = activation
        self.dense_1 = nn.Linear(units, units)
        self.dense_2 = nn.Linear(units, units)

        self.reset_params()

    def reset_params(self):
        GlorotOrthogonal(self.dense_1.weight)
        nn.init.zeros_(self.dense_1.bias)
        GlorotOrthogonal(self.dense_2.weight)
        nn.init.zeros_(self.dense_2.bias)

    def forward(self, inputs):
        x = self.dense_1(inputs)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dense_2(x)
        if self.activation is not None:
            x = self.activation(x)
        return inputs + x
