import torch
import torch.nn as nn
import torch.nn.functional as F

class TnetBatch(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        # input.shape == bs x [3,n]
        bs = len(input)
        xb = [F.relu(self.bn1(self.conv1(x).unsqueeze(0))) for x in input]
        xb = [F.relu(self.bn2(self.conv2(x))) for x in xb]
        xb = [F.relu(self.bn3(self.conv3(x))) for x in xb]
        pool = [F.max_pool1d(x, x.size(-1)).flatten() for x in xb]
        h = torch.stack(pool) # [bs, 1024]
        xb = F.relu(self.bn4(self.fc1(h)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1).to(input[0].device)
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init # [bs, k*k] -> [bs, k, k]
        return matrix

class TransformBatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = TnetBatch(k=3)
        self.feature_transform = TnetBatch(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        # input is a list [bs, 3, n]
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = [torch.mm(pcd.t(), rm).t() for pcd, rm in zip(input, matrix3x3)] # [bs, 3, n]
        #xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = [F.relu(self.bn1(self.conv1(x.unsqueeze(0)))).squeeze() for x in xb]

        matrix64x64 = self.feature_transform(xb)
        xb = [torch.mm(pcd.t(), rm).t() for pcd, rm in zip(xb, matrix64x64)] # [bs, 64, n]
        #xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = [F.relu(self.bn2(self.conv2(x.unsqueeze(0)))) for x in xb]
        xb = [self.bn3(self.conv3(x)) for x in xb]
        output = torch.stack([F.max_pool1d(x, x.size(-1)).flatten() for x in xb])
        return output, matrix3x3, matrix64x64


class PointNetBatch(nn.Module):
    def __init__(self, output=1):
        super().__init__()
        self.transform = TransformBatch()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        #self.logsoftmax = nn.LogSoftmax(dim=1)
        #todo : are loss

    def forward(self, input):
        # actually input is a list : [bs, 3, n]

        h, matrix3x3, matrix64x64 = self.transform(input)
        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.dropout(self.fc2(h))))
        output = self.fc3(h)
        return output, matrix3x3, matrix64x64

        #return self.logsoftmax(output), matrix3x3, matrix64x64



class TnetBatchNB(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

    def forward(self, input):
        # input.shape == bs x [3,n]
        bs = len(input)
        xb = [F.relu(self.conv1(x).unsqueeze(0)) for x in input]
        xb = [F.relu(self.conv2(x)) for x in xb]
        xb = [F.relu(self.conv3(x)) for x in xb]
        pool = [F.max_pool1d(x, x.size(-1)).flatten() for x in xb]
        h = torch.stack(pool) # [bs, 1024]
        xb = F.relu(self.fc1(h))
        xb = F.relu(self.fc2(xb))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1).to(input[0].device)
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init # [bs, k*k] -> [bs, k, k]
        return matrix

class TransformBatchNB(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = TnetBatchNB(k=3)
        self.feature_transform = TnetBatchNB(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

    def forward(self, input):
        # input is a list [bs, 3, n]
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = [torch.mm(pcd.t(), rm).t() for pcd, rm in zip(input, matrix3x3)] # [bs, 3, n]
        #xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = [F.relu(self.conv1(x.unsqueeze(0))).squeeze() for x in xb]

        matrix64x64 = self.feature_transform(xb)
        xb = [torch.mm(pcd.t(), rm).t() for pcd, rm in zip(xb, matrix64x64)] # [bs, 64, n]
        #xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = [F.relu(self.conv2(x.unsqueeze(0))) for x in xb]
        xb = [self.conv3(x) for x in xb]
        output = torch.stack([F.max_pool1d(x, x.size(-1)).flatten() for x in xb])
        return output, matrix3x3, matrix64x64


class PointNetBatchNB(nn.Module):
    def __init__(self, output=1):
        super().__init__()
        self.transform = TransformBatchNB()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output)

        self.dropout = nn.Dropout(p=0.3)
        #self.logsoftmax = nn.LogSoftmax(dim=1)
        #todo : are loss

    def forward(self, input):
        # actually input is a list : [bs, 3, n]

        h, matrix3x3, matrix64x64 = self.transform(input)
        h = F.relu(self.fc1(h))
        h = F.relu(self.dropout(self.fc2(h)))
        output = self.fc3(h)
        return output, matrix3x3, matrix64x64

        #return self.logsoftmax(output), matrix3x3, matrix64x64



class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        # input.shape == (k, n)
        print(input.shape)
        h = F.relu(self.bn1(self.conv1(input))) # [64, n]
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h))) # [1024, n]
        h = F.max_pool1d(h, h.size(-1)).flatten()
        h = F.relu(self.bn4(self.fc1(h)))
        h = F.relu(self.bn5(self.fc2(h)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).to(input.device)

        matrix = self.fc3(h).view(self.k,self.k) + init # [k*k] -> [k, k]
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        # input is a list [bs, 3, n]
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.mm(input.t(), matrix3x3).t()  # [3, n]
        #xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.mm(xb.t(), matrix64x64).t() # [64, n]
        #xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        output = F.max_pool1d(xb, xb.size(-1)).flatten()
        return output, matrix3x3, matrix64x64


class PointNet(nn.Module):
    def __init__(self, output=1):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        #self.logsoftmax = nn.LogSoftmax(dim=1)
        #todo : are loss

    def forward(self, input):
        # input [3, n]

        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return output, matrix3x3, matrix64x64

        #return self.logsoftmax(output), matrix3x3, matrix64x64

