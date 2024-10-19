import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple

from torchvision.models import resnet18, resnet50
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import pretrainedmodels
import ssl
ssl._create_default_https_context = ssl._create_unverified_context # for pretrainedmodels

class MKGE(nn.Module):
    def __init__(self, args, num_ent_uid, target_list, device, all_locs=None, num_habitat=None, all_timestamps=None, all_loc_times=None):
        super(MKGE, self).__init__()
        self.args = args
        self.num_ent_uid = num_ent_uid

        self.ent_embedding = torch.nn.Embedding(self.num_ent_uid, args.embedding_dim, sparse=False)

        if self.args.use_learned_loc_embed:
            self.location_embedding = torch.nn.Embedding(len(all_locs), args.embedding_dim)
        else:
            self.location_embedding = MLP(args.location_input_dim, args.embedding_dim, args.mlp_location_numlayer)

        self.time_embedding = MLP(args.time_input_dim, args.embedding_dim, args.mlp_time_numlayer)
        # print(self.time_embedding)
        # print(self.location_embedding)
        
        if self.args.img_embed_model == 'resnet50':
            self.image_embedding = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.image_embedding.fc = nn.Linear(2048, args.embedding_dim)
        elif self.args.img_embed_model == 'resnet18':
            self.image_embedding = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.image_embedding.fc = nn.Linear(512, args.embedding_dim)
            # self.image_embedding.fc = nn.Linear(512, 182)
        else:
            raise NotImplementedError

        self.target_list = target_list
        if all_locs is not None:
            self.all_locs = all_locs.to(device)
        if all_timestamps is not None:
            self.all_timestamps = all_timestamps.to(device)
        if all_loc_times is not None:
            self.all_loc_times = all_loc_times.to(device)

        #print(self.all_locs)

        if self.args.add_inverse_rels:
            num_relations = 4
        else:
            num_relations = 2

        self.act = nn.PReLU()
        
        self.mlp = nn.Linear(3*args.embedding_dim, args.embedding_dim)
        self.layer_norm = nn.LayerNorm(3*args.embedding_dim)

        self.classifier = nn.Linear(args.embedding_dim, len(self.target_list))

        self.args = args
        self.device = device

        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        # nn.init.xavier_uniform_(self.rel_embedding.weight.data)

        if self.args.img_embed_model in ['resnet18', 'resnet50']:
            nn.init.xavier_uniform_(self.image_embedding.fc.weight.data)

        if self.args.use_learned_loc_embed:
            nn.init.xavier_uniform_(self.location_embedding.weight.data)

        nn.init.xavier_uniform_(self.mlp.weight.data)

        nn.init.xavier_uniform_(self.classifier.weight.data)

    # @profile
    def forward_ce(self, graph, image, time, location=None):

        # create a graph using location and time attributes of the image
        # print('graph.n_id = {}'.format(graph.n_id))

        # node ids:
        # <image>: 0
        # T: 1
        # L: 2

        # edge ids:
        # (<image>, T): 0
        # (<image>, L): 1

        # gather initial node embedding
        batch_size = image.size(0)

        img_embed = self.image_embedding(image)

        # print('img_embed = {}'.format(img_embed))

        time_emb = self.time_embedding(time)

        if location is not None:
            loc_emb = self.location_embedding(location)

        if location is not None:
            node_emb = torch.stack([img_embed, time_emb, loc_emb], dim=1) # [batch, n_nodes, hid_dim]
        else:
            node_emb = torch.stack([img_embed, time_emb], dim=1)

        node_emb = node_emb.view(node_emb.size(0), -1)
        
        node_emb = self.layer_norm(node_emb)

        img_context_emb = self.mlp(node_emb)

        img_context_emb = self.act(img_context_emb)

        # project the embeddding using a linear layer to compute label distribution
        score = self.classifier(img_context_emb)
        # print('score = {}'.format(score.size()))

        return score


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=3,
                 p_dropout=0.0,
                 bias=True):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.p_dropout = p_dropout
        step_size = (input_dim - output_dim) // num_layers
        hidden_dims = [output_dim + (i * step_size)
                       for i in reversed(range(num_layers))]

        mlp = list()
        layer_indim = input_dim
        for hidden_dim in hidden_dims:
            mlp.extend([nn.Linear(layer_indim, hidden_dim, bias),
                        nn.Dropout(p=self.p_dropout, inplace=True),
                        nn.PReLU()])

            layer_indim = hidden_dim

        self.mlp = nn.Sequential(*mlp)

        # initiate weights
        self.init()

    def forward(self, x):
        return self.mlp(x)

    def init(self):
        for param in self.parameters():
            nn.init.uniform_(param)
