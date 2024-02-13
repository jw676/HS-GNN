import torch.nn as nn

class LinearRgressor(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers
                 ):
        super(LinearRgressor, self).__init__()

        self.linears = nn.ModuleList()

        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]

        for input, output in zip(dims[:-1], dims[1:]):
            self.linears.append(nn.Linear(input, output))

        self.activation = nn.ReLU()

    def forward(self, data):
        h = data
        for linear in self.linears[:-1]:
            h = linear(h)
            h = self.activation(h)
        h = self.linears[-1](h)

        return h


class MLPG(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 add_layer
                 ):
        super(MLPG, self).__init__()

        self.linears = nn.ModuleList()

        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]

        for input, output in zip(dims[:-1], dims[1:]):
            self.linears.append(nn.Linear(input, output))

        self.activation = nn.ReLU()

        self.add_layer = add_layer

    def forward(self, data, add_data=None):
        h = data

        if add_data is None:
            for idx, linear in enumerate(self.linears[:-1]):
                h = linear(h)
                h = self.activation(h)
            h = self.linears[-1](h)

            return h

        if isinstance(add_data, list):
            h_add = add_data[0] + add_data[1]
        else:
            h_add = add_data
        for idx, linear in enumerate(self.linears[:-1]):
            if idx==self.add_layer:
                h = h + h_add
            h = linear(h)
            h = self.activation(h)
        h = self.linears[-1](h)

        return h


class MLPR(nn.Module):

    def __init__(self,
                 in_dim,
                 add_dim,
                 hid_dim,
                 out_dim,
                 num_layers
                 ):
        super(MLPR, self).__init__()



        dims =  [hid_dim] * (num_layers - 1) + [out_dim]

        self.flinear = nn.Linear(in_dim, hid_dim)
        self.alinear = nn.Linear(add_dim, hid_dim)

        self.linears = nn.ModuleList()

        for input, output in zip(dims[:-1], dims[1:]):
            self.linears.append(nn.Linear(input, output))

        self.activation = nn.ReLU()


    def forward(self, data, add_data):
        h_f = self.activation(self.flinear(data))
        h_a = self.activation(self.alinear(add_data))

        h = h_f * h_a

        for linear in self.linears[:-1]:
            h = linear(h)

        h = self.linears[-1](h)

        return h
