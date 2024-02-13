import torch.nn as nn

class MLP_Layer(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 out_dim,
                 hid_dim=None,
                 batch_norm=True):
        '''
        :param num_layers:
        :param in_dim:
        :param out_dim:
        :param hid_dim:
        :param batch_norm:
        :return:
        '''
        super().__init__()
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        dims = [in_dim] + [hid_dim for _ in range(num_layers-1)] + [out_dim]

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.linears.append(nn.Linear(d_in, d_out))

        if batch_norm:
            for _ in range(num_layers-1):
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))

        self.relu = nn.ReLU()

        self.norm = batch_norm

    def forward(self, h):

        h = self.linears[0](h)

        if self.norm:
            for batch_norm, linear in zip(self.batch_norms, self.linears[1:]):
                h = self.relu(h)
                h = batch_norm(h)
                h = linear(h)

        else:
            for linear in self.linears[1:]:
                h = self.relu(h)
                h = linear(h)

        return h
