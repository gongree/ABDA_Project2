import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# extractor
class ConvKRegion(nn.Module):
    def __init__(self, k=1, out_size=8, kernel_size=8, stride_size=2, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=stride_size)

        output_dim_1 = (time_series - kernel_size) // stride_size + 1

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                            kernel_size=kernel_size)
        output_dim_2 = (output_dim_1 - kernel_size)//(stride_size//2) + 1
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16,
                            kernel_size=kernel_size)
        output_dim_3 = (output_dim_2 - kernel_size)//(stride_size//2) + 1
        self.max_pool1 = nn.MaxPool1d(pool_size)
        output_dim_4 = (output_dim_3 // pool_size) * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            nn.Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape
        x = torch.transpose(x, 1, 2)
        x = self.in0(x)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view((b*k, 1, d))
        x = self.conv1(x)
        x = self.in1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.conv3(x)
        x = self.in3(x)
        x = self.max_pool1(x)
        x = x.view((b, k, -1))
        x = self.linear(x)

        return x

class GruKRegion(nn.Module):
    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(kernel_size, out_size)
        )

    def forward(self, bold):

        b, n, _ = bold.shape
        x = bold.view((b*n, -1, self.kernel_size))
        x, _ = self.gru(x)
        x = x[:, -1, :].view((b, n, -1))
        x = self.linear(x)
        return x

# Embed2Graph
class Embed2GraphByProduct(nn.Module):
    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)
        m = torch.unsqueeze(m, -1)

        return m
    

class Embed2GraphByLinear(nn.Module):
    def __init__(self, input_dim, roi_num=122, device='cpu'):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        identity = torch.eye(roi_num, device=device)  # ROI 수만큼 단위 행렬 생성
        off_diag = torch.ones(roi_num, roi_num, device=device) - torch.eye(roi_num, device=device)  # 대각 제외
        self.rel_rec = identity.repeat(roi_num, 1) * off_diag.flatten()[:, None]
        self.rel_send = identity.repeat_interleave(roi_num, dim=0) * off_diag.flatten()[:, None]

    def forward(self, x):

        b, n, _ = x.shape
        receivers = torch.einsum('ij,bjk->bik', self.rel_rec, x)
        senders = torch.einsum('ij,bjk->bik', self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)

        x = torch.relu(self.fc_out(x))
        x = torch.relu(self.fc_cat(x))

        m = torch.reshape(
            x, (b, n, n, -1))
        return m

# GCN
class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear(8*roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )
    
    def forward(self, node, edge):
        b = node.shape[0]

        x = torch.einsum('ijk,ijp->ijp', edge, node)
        x = self.gcn(x)
        x = x.reshape((b*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((b, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', edge, x)
        x = self.gcn1(x)
        x = x.reshape((b*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((b, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', edge, x)
        x = self.gcn2(x)
        x = self.bn3(x)
        x = x.view(b,-1)

        return self.fcn(x)

# main model
class FBNETGEN(nn.Module):
    def __init__(self, num_node=112, node_feature_dim=112, embed_dim=8, time_length=232,extractor_type='cnn', device='cpu'):
        super().__init__()

        self.graph_generation = "product" # linear or product
        self.extractor_type = extractor_type # cnn or gru

        if self.extractor_type == "cnn":
            self.extract = ConvKRegion(
                out_size=embed_dim,
                kernel_size=4, # window size
                stride_size=2,
                pool_size=16,
                time_series=time_length
            )
        elif self.extractor_type == 'gru':
            self.extract = GruKRegion(
                out_size=embed_dim,
                kernel_size=4,
                layers=4
            )

        if self.graph_generation == 'linear':
            self.emb2graph = Embed2GraphByLinear(
                embed_dim, roi_num=num_node, device=device
            )
        elif self.graph_generation == 'product':
            self.emb2graph = Embed2GraphByProduct(
                embed_dim, roi_num=num_node
            )
        
        self.linear = nn.Linear(time_length, embed_dim)
        self.predictor = GNNPredictor(node_feature_dim, roi_num=num_node)

    def forward(self, bold, pcc):
        x = self.extract(bold)
        x = F.softmax(x, dim=-1)

        edge = self.emb2graph(x)

        edge = edge[:, :, :, 0]

        b, _, _ = edge.shape

        # edge_variance = torch.mean(torch.var(edge.reshape((b, -1)), dim=1))
        logit = self.predictor(pcc, edge)

        return logit, edge #, edge_variance