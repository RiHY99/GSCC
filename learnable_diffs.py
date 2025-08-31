import torch,os
from torch import nn
from torch.nn import functional as F
import numpy as np
import sys
import json
import torch_geometric as tg
from torch_geometric.nn import GCNConv
from my_funcs import *




class CrossAtten(nn.Module):
    def __init__(self, fusion_dropout, cross_atten_dim, fusion_n_head):
        super(CrossAtten, self).__init__()
        self.attention = nn.MultiheadAttention(cross_atten_dim, fusion_n_head, dropout=fusion_dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(cross_atten_dim)
        self.norm2 = nn.LayerNorm(cross_atten_dim)

        self.dropout1 = nn.Dropout(fusion_dropout)
        self.dropout2 = nn.Dropout(fusion_dropout)
        self.dropout3 = nn.Dropout(fusion_dropout)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(cross_atten_dim, cross_atten_dim * 4)
        self.linear2 = nn.Linear(cross_atten_dim * 4, cross_atten_dim)

    def forward(self, input1, input2):
        output_1, attn_weight = self.cross(input1, input2)

        return output_1, attn_weight
    def cross(self, q, kv, attn_mask=None):
        attn_output, attn_weight = self.attention(q, kv, kv, attn_mask=None)

        output = q + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output, attn_weight



class ChangeFilter(nn.Module):
    def __init__(self, args, ini_d, self_loops=True):
        super(ChangeFilter, self).__init__()
        if args.encoder == 'resnet':
            self.mode = 'cnn'
        elif args.encoder == 'transformer':
            self.mode = 'trans'
        else:
            assert False, 'Unknown Encoder'
        self.args = args
        self.ini_d = ini_d
        self.out_d = args.graph_out_d

        self.proj = nn.Linear(ini_d, args.graph_d)

        self.gcn = GCNConv(args.graph_d, args.graph_out_d, add_self_loops=self_loops)
        self.layer_N = nn.LayerNorm(args.graph_out_d)
        # self.act = nn.ReLU()
        self.act = nn.GELU()

        self.diff_query = nn.Parameter(torch.randn(1, 1, ini_d))
        nn.init.xavier_normal_(self.diff_query)

        self.linear_q = nn.Linear(args.graph_out_d, args.graph_out_d)
        self.linear_k = nn.Linear(args.graph_out_d, args.graph_out_d)


    def forward(self, img, all_edge):
        batch = img.shape[0]
        channel = img.shape[1]
        h = img.shape[2]
        w = img.shape[3]
        node_num = h * w
        batch_diff_query = self.diff_query.expand(batch, -1, -1)

        if self.mode == 'trans':
            # # [batch, token+1, dim] to [batch*(token+1), dim]
            # img = img.view(-1, img.shape[-1])
            raise NotImplementedError
        elif self.mode == 'cnn':
            # [batch, channel, h, w] to [batch*h*w, channel]
            channel = img.shape[1]
            # [batch, h * w, channel]
            img = img.transpose(1, 2).transpose(2, 3).contiguous().view(batch, -1, channel)
        else:
            raise NotImplementedError

        # [batch, h * w, channel]
        img = self.proj(img)

        img = img.view(-1, channel)

        img_graph = self.gcn(img, all_edge)
        img_graph = self.act(img_graph)

        img_graph = img_graph + img
        img_graph = self.layer_N(img_graph)

        self_adj = self.get_adj(batch_diff_query, img_graph, batch, node_num, self.out_d)

        return img_graph, self_adj

    def get_adj(self, batch_diff_query, img, batch, node_num, out_d, temperature=0.1):
        # [batch, 1, dim] to [batch*8, 1, dim/8]
        batch_diff_query = batch_diff_query.transpose(0, 1).contiguous().view(1, batch * self.args.mask_head, int(self.args.graph_out_d / self.args.mask_head)).transpose(0, 1)
        # [batch, token, dim]
        img = img.view(batch, node_num, out_d).transpose(0, 1).contiguous()
        # [batch*8, token, dim/8]
        img = img.view(node_num, batch * self.args.mask_head, int(self.args.graph_out_d / self.args.mask_head)).transpose(0, 1).contiguous()

        adj = torch.bmm(batch_diff_query, img.transpose(-2, -1))
        adj = F.softmax(adj / temperature, dim=-1)

        return adj



class myresblock_ori_G(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(myresblock_ori_G, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel/2), kernel_size=1),
            nn.BatchNorm2d(int(outchannel/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel/2), int(outchannel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(outchannel / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel/2),outchannel,kernel_size=1),
            nn.BatchNorm2d(outchannel)
        )

        if inchannel == outchannel:
            self.plus = 'straight'
        else:
            self.plus = 'linear'
            self.linear = nn.Linear(in_features=inchannel, out_features=outchannel)

    def forward(self, x):
        out = self.left(x)
        if self.plus == 'straight':
            residual = x
        else:
            residual = self.linear(x)
        out += residual
        return F.relu(out)


class FeatureFusion_ori_G(nn.Module):
    def __init__(self, args, ini_dim, cross_atten_dim):
        super(FeatureFusion_ori_G, self).__init__()
        self.args = args
        self.cross_atten_dim = cross_atten_dim
        self.proj = nn.Linear(ini_dim, cross_atten_dim)
        self.self_atten = CrossAtten(args.fusion_dropout, cross_atten_dim, args.fusion_n_head)
        self.fusion_transformer = nn.ModuleList([CrossAtten(args.fusion_dropout, cross_atten_dim, args.fusion_n_head) for i in range(args.fusion_n_layers)])
        if args.encoder == 'resnet':
            self.position_embeddings = nn.Embedding(1000, cross_atten_dim)
        elif args.encoder == 'transformer':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.resblock = nn.ModuleList([myresblock_ori_G(cross_atten_dim * 2, cross_atten_dim * 2) for i in range(args.res_n_layers)])

        self.LN = nn.ModuleList([nn.LayerNorm(cross_atten_dim * 2) for i in range(args.res_n_layers)])

    def forward(self, batch, x1, x2):
        # batch = x1.shape[0]
        if self.args.encoder == 'resnet':
            channel = x1.shape[1]
            x1 = x1.view(batch, channel, -1).permute(0, 2, 1)  # [batch_size, h*w, dim]
            x2 = x2.view(batch, channel, -1).permute(0, 2, 1)
        elif self.args.encoder == 'transformer':
            raise NotImplementedError
        else:
            assert False, 'Unknown Encoder'

        x1 = self.proj(x1)
        x2 = self.proj(x2)

        if self.args.encoder == 'resnet':
            length = x1.shape[1]
            positions = torch.arange(length, device=x1.device).expand(batch, length)
            position_embeddings = self.position_embeddings(positions)  # [batch, length, dim]
            x1 = x1 + position_embeddings
            x2 = x2 + position_embeddings

        output1_list = []
        weight1_list = []
        output2_list = []
        weight2_list = []
        x1_, _ = self.self_atten(x1, x1)
        x2_, _ = self.self_atten(x2, x2)
        x1 = x1_ + x1
        x2 = x2_ + x2
        query1 = x1
        query2 = x2
        kv = x2 - x1
        for layer in self.fusion_transformer:
            query1, attn_weight1 = layer(query1, kv)
            query2, attn_weight2 = layer(query2, kv)
            if self.args.kv_new:
                kv = query2 - query1
            else:
                pass
            #
            output1_list.append(query1)
            weight1_list.append(attn_weight1)
            output2_list.append(query2)
            weight2_list.append(attn_weight2)

        i = 0
        output = torch.zeros((batch, length, self.cross_atten_dim * 2)).to(x1.device)
        for res in self.resblock:
            # [batch, length, channel]
            input = torch.cat([output1_list[i], output2_list[i]], dim=-1)
            output = output + input

            output = output.permute(0, 2, 1).view(batch, self.cross_atten_dim * 2, 14, 14)  # batch*1024*14*14
            output = res(output)
            output = output.view(batch, self.cross_atten_dim * 2, -1).permute(0, 2, 1)  # batch*(14*14)*1024

            output = self.LN[i](output)
            i = i + 1

        return output


class CompatibilityModule:
    def __init__(self):
        self.CrossAtten = CrossAtten

def enable_backward_compatibility():
    sys.modules['feature_fusion'] = CompatibilityModule()

def disable_backward_compatibility():
    if 'feature_fusion' in sys.modules and isinstance(sys.modules['feature_fusion'], CompatibilityModule):
        del sys.modules['feature_fusion']

def load_compatible_model(checkpoint_path, device):
    
    enable_backward_compatibility()
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        
        return checkpoint
        
    finally:
        disable_backward_compatibility()





