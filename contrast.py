import torch
import torch.nn as nn

class SSL(nn.Module):
    def __init__(self, model1, model2, num_input):
        super(SSL, self).__init__()

        self.model1 = model1
        self.model2 = model2

        self.bilinear = nn.Bilinear(num_input, num_input, 1)

    def forward(self, graph1, graph2):

        output1 = self.model1(graph1)
        output2 = self.model2(graph2)

        logits = self.bilinear(output1, output2) # [Batch, 1]

        return logits

    def loss(self, graph1, graph2):

        output1, output2 = self.forward(graph1, graph2)

        return torch.mean(torch.abs(torch.sigmoid(output1)-torch.sigmoid(output2)), dim=-1)
