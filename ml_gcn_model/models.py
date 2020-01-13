import torchvision.models as models
from torch.nn import Parameter
from ml_gcn_model.util import *
import torch
import torch.nn as nn
import pickle

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        inp = inp[0]
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]

def gcn_resnet101(num_classes, t, pretrained=True, adj_file=None, in_channel=300):
    model = models.resnet101()
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)


class Model(nn.Module):

    def __init__(self, model, inp):
        super(Model, self).__init__()
        self.model = model
        self.inp = Parameter(torch.from_numpy(np.expand_dims(inp, axis=0)),requires_grad=False)
        self.std = Parameter(torch.tensor([0.229, 0.224, 0.225]), requires_grad=False)
        self.mean = Parameter(torch.tensor([0.485, 0.456, 0.406]), requires_grad=False)


    def forward(self, feature):
        '''
        :param feature:
          value in [0,1]
          size [batch, channels, height, weight]
        :return:
        '''
        x = feature.permute(0, 2, 3, 1) - self.mean
        x = x / self.std
        x = x.permute(0, 3, 1, 2)
        x = self.model(x, self.inp)
        x = torch.sigmoid(x)
        return x

def gcn_resnet101_attack(num_classes, t, pretrained=True, adj_file=None, word_vec_file=None, save_model_path=None, in_channel=300):
    model = gcn_resnet101(num_classes, t, pretrained, adj_file, in_channel=300)
    with open(word_vec_file, 'rb') as f:
        inp = pickle.load(f)
    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = Model(model, inp)
    return model

