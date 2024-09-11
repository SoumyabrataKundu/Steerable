import torch
import torch.nn as nn
from numpy import prod, sqrt
from Steerable.nn.Steerable3d.Steerable3d_pytorch.conv_layers import SE3BatchNorm, SE3NormNonLinearity
from Steerable.nn.Steerable3d.utils import merge_channel_dim, get_pos_encod, split_channel_dim

#######################################################################################################################
############################################ SE(3) Multihead Self Attention ###########################################
#######################################################################################################################

class SE3MultiSelfAttention(nn.Module):
    def __init__(self, transformer_dim, n_head, add_pos_enc=True):
        super(SE3MultiSelfAttention, self).__init__()

        self.query_dim = []
        self.n_head = n_head
        self.add_pos_enc = add_pos_enc
        self.transformer_dim = [transformer_dim] if type(transformer_dim) is not list and type(transformer_dim) is not tuple else transformer_dim
        self.maxl = len(self.transformer_dim) - 1
        self.scale = sqrt(sum([(2*l+1)*dim for l, dim in enumerate(self.transformer_dim)]))
        for dim in self.transformer_dim:
            if not dim % n_head == 0 :
                raise ValueError(f"Transformer dimension ({dim}) is not divisible by number of heads ({n_head}).")
            self.query_dim.append(dim // n_head)

        # Layer Parameters
        self.embeddings = nn.ParameterList([nn.Parameter(
                                        torch.randn(3, 1, 1, dim, dim, dtype = torch.cfloat))
                                        for dim in self.transformer_dim])
        self.encoding = nn.ParameterList([nn.Parameter(
                                        torch.randn(2, n_head, 1, 1, dim, 1, dtype = torch.cfloat))
                                        for dim in self.query_dim])
        # self.encoding1 = nn.ParameterList([nn.Parameter(
        #                                 torch.randn(n_head, 1, 1, dim, 1, dtype = torch.cfloat))
        #                                 for dim in self.query_dim])
        # self.encoding2 = nn.ParameterList([nn.Parameter(
        #                                 torch.randn(n_head, 1, 1, dim, 1, dtype = torch.cfloat))
        #                                 for dim in self.query_dim])
        self.out = nn.ParameterList([nn.Parameter(
                                        torch.randn(dim, dim, dtype = torch.cfloat))
                                        for dim in self.transformer_dim])

        self.pos_encod = None
        
    # def forward(self, x):
    #     x_shape = x[0].shape
        
    #     if self.pos_encod is None and self.add_pos_enc:
    #         self.pos_encod = get_pos_encod(x_shape[-3:], self.maxl)
    #         self.pos_encod = [part.to(x[0].device) for part in self.pos_encod]

    #     # Query Key Pair
    #     A = torch.zeros(x_shape[0], self.n_head, *[prod(x_shape[3:])]*2, dtype=torch.cfloat, device = x[0].device)
    #     B = []
    #     for l in range(self.maxl+1):
    #         # Embeddings
    #         E = (self.embeddings[l] @ x[l].flatten(3))
    #         E = E.reshape(3, x_shape[0], (2*l+1), self.n_head, self.query_dim[l], -1).transpose(2,3).flatten(3,4)
    #         Q, K = torch.conj(E[0].transpose(-2,-1)), E[1]
    #         B.append(E[2])

    #         # Attention Scores
    #         if self.add_pos_enc:
    #             pos1 = (self.encoding1[l] * self.pos_encod[l]).flatten(2,3)
    #             A = A + (Q @ K + (Q.unsqueeze(-2) @ pos1).squeeze(-2)) / sqrt(self.query_dim[l])
    #         else:
    #             A = A + (Q @ K) / sqrt(self.query_dim[l])
            
    #     A = nn.functional.softmax(A.abs(), dim=-1).type(torch.cfloat)
        
    #     # Output
    #     result = []
    #     for l in range(self.maxl+1):
    #         V = B[l] @ A.transpose(-2,-1)
    #         if self.add_pos_enc:
    #             pos2 = (self.encoding2[l] * self.pos_encod[l]).flatten(2,3)
    #             V = V + (pos2 @ A.unsqueeze(-1)).squeeze(-1).transpose(-2,-1)
                
    #         V = self.out[l] @ V.reshape(x_shape[0], self.n_head, 2*l+1, self.query_dim[l], -1).transpose(1,2).flatten(2,3)
    #         result.append(V.reshape(x_shape[0], 2*l+1, self.transformer_dim[l], *x_shape[3:]))

    #     return result
    
    def forward(self, x):
        x_shape = x[0].shape
        
        if self.pos_encod is None and self.add_pos_enc:
            self.pos_encod = get_pos_encod(x_shape[-3:], self.maxl)
            self.pos_encod = [part.to(x[0].device) for part in self.pos_encod]

        # Query Key Pair
        E, P = [], []
        for l in range(self.maxl+1):
            # Embeddings
            E.append((self.embeddings[l] @ x[l].flatten(3)).reshape(3, x_shape[0], (2*l+1), self.n_head, self.query_dim[l], -1).transpose(2,3).flatten(3,4))
            # Attention Scores
            if self.add_pos_enc:
                P.append((self.encoding[l] * self.pos_encod[l]).flatten(3,4))
        
        QKV = torch.cat(E, dim=3)
        Q, K, V = torch.conj(QKV[0].transpose(-2,-1)), QKV[1], QKV[2]
        A = (Q @ K) / self.scale
        if self.add_pos_enc:
            pos = torch.cat(P, dim=-2)
            A = A + (Q.unsqueeze(-2) @ pos[0]).squeeze(-2) / self.scale
        A = nn.functional.softmax(A.abs(), dim=-1).type(torch.cfloat)
        
        V = V @ A.transpose(-2,-1)
        if self.add_pos_enc:
            V = V + (pos[1] @ A.unsqueeze(-1)).squeeze(-1).transpose(-2,-1)
        V = split_channel_dim(V.transpose(1,2), self.transformer_dim)
        
        # Output
        result = [(self.out[l] @ V[l].reshape(x_shape[0], 2*l+1, self.transformer_dim[l], -1)).reshape(*x[l].shape)
                  for l in range(self.maxl+1)]

        return result

#######################################################################################################################
################################################### SE(3) Transformer #################################################
#######################################################################################################################

class PositionwiseFeedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionwiseFeedforward, self).__init__()

        self.input_dim = [input_dim] if type(input_dim) is not list and type(input_dim) is not tuple else input_dim
        self.hidden_dim = [hidden_dim] if type(hidden_dim) is not list and type(hidden_dim) is not tuple else hidden_dim
        
        assert len(self.hidden_dim) == len(self.input_dim)
        
        self.weights1 = nn.ParameterList([nn.Parameter(
                                        torch.randn(hidden, input, dtype = torch.cfloat))
                                         for input, hidden in zip(self.input_dim, self.hidden_dim)])
        self.nonlinearity = SE3NormNonLinearity(hidden_dim)

        self.weights2 = nn.ParameterList([nn.Parameter(
                                        torch.randn(input, hidden, dtype = torch.cfloat))
                                         for input, hidden in zip(self.input_dim, self.hidden_dim)])

    def forward(self, x):
        x_shape = x[0].shape
        x = [(self.weights1[l] @ part.flatten(3)).reshape(x_shape[0], 2*l+1, -1, *x_shape[3:]) for l,part in enumerate(x)]
        x = self.nonlinearity(x)
        x = [(self.weights2[l] @ part.flatten(3)).reshape(x_shape[0], 2*l+1, -1, *x_shape[3:]) for l,part in enumerate(x)]
         
        return x
    
#######################################################################################################################
################################################### SE(3) Transformer #################################################
#######################################################################################################################

class SE3Transformer(nn.Module):
    def __init__(self, transformer_dim, n_head, hidden_dim, add_pos_enc=True):
        super(SE3Transformer, self).__init__()

        # Layer Design
        self.multihead_attention = SE3MultiSelfAttention(transformer_dim, n_head, add_pos_enc=add_pos_enc)
        self.positionwise_feedforward = PositionwiseFeedforward(transformer_dim, hidden_dim)

        self.layer_norm1 = SE3BatchNorm()
        self.layer_norm2 = SE3BatchNorm()

    def forward(self, x):
        attentions = self.multihead_attention(self.layer_norm1(x))
        x = [attention + part for attention, part in zip(attentions, x)]
        postions = self.positionwise_feedforward(self.layer_norm2(x))
        x = [postion + part for postion, part in zip(postions, x)]

        return x

class SE3TransformerEncoder(nn.Module):
    def __init__(self, transformer_dim, n_head, n_layers = 1, add_pos_enc=True):
        super(SE3TransformerEncoder, self).__init__()

        # Layer Design
        self.transformer_encoder = torch.nn.Sequential(
            *[SE3Transformer(transformer_dim, n_head, transformer_dim, add_pos_enc=add_pos_enc) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        
        return x
    
    
#######################################################################################################################
############################################## SE(3) Transformer Decoder ##############################################
####################################################################################################################### 

class SE3TransformerDecoder(nn.Module):
    def __init__(self, transformer_dim, n_head, n_classes, n_layers, add_pos_enc=True):
        super(SE3TransformerDecoder, self).__init__()

        self.transformer_dim = transformer_dim
        self.n_classes = n_classes
        self.maxl = len(self.transformer_dim) - 1
        self.transformer_encoder = torch.nn.ModuleList(
            [SE3Transformer(transformer_dim, n_head, transformer_dim, add_pos_enc=add_pos_enc) for _ in range(n_layers)]
        )

        self.class_embed = nn.Parameter(torch.randn(1, 1, transformer_dim[0], n_classes, dtype=torch.cfloat))
        self.pos_encod = None
        self.norm = SE3BatchNorm()
        
    def forward(self, x):
        if self.pos_encod == None:
            pos = get_pos_encod(x[0].shape[-3:], self.maxl)
            pos = [part.to(x[0].device) for part in pos]
            self.pos_encod = [torch.zeros(part.shape[0]+self.n_classes, part.shape[1], 1, part.shape[3]+self.n_classes, 
                                          dtype=torch.cfloat, device=x[0].device) for part in pos]
            for l in range(len(pos)):
                self.pos_encod[l][:pos[l].shape[0], :,:,:pos[l].shape[3]] = pos[l]
            for module in self.transformer_encoder:
                module.multihead_attention.pos_encod = self.pos_encod
                
        
        result = []
        x_shape = x[0].shape
        for l, part in enumerate(x):
            if l==0:
                class_embed = self.class_embed.expand(x[0].shape[0], -1, -1, -1)
            else:
                class_embed = torch.zeros(*part.shape[:3], self.n_classes, dtype=torch.cfloat, device=part.device)
            
            result.append(torch.cat([part.flatten(3), class_embed], dim=-1))
        
        for module in self.transformer_encoder:
            result = module(result)
        result = self.norm(result)
        
        patches, cls_seg_feat = [], []
        for l, part in enumerate(result):
            patches.append(part[..., : -self.n_classes].reshape(x_shape[0], 2*l+1, -1, *x_shape[3:]))
            cls_seg_feat.append(part[..., -self.n_classes :])
        
  
        return patches, cls_seg_feat
    
class SE3ClassEmbedings(nn.Module):
    def __init__(self, transformer_dim, embedding_dim):
        super(SE3ClassEmbedings, self).__init__()
        
        self.transformer_dim = [transformer_dim] if type(transformer_dim) is not list and type(transformer_dim) is not tuple else transformer_dim
        self.embedding_dim = [embedding_dim] if type(embedding_dim) is not list and type(embedding_dim) is not tuple else embedding_dim
        
        assert len(self.transformer_dim) == len(self.embedding_dim)
        
        self.weight = nn.ParameterList([
            nn.Parameter(torch.randn(dim1, dim2, dtype=torch.cfloat)) for dim1, dim2 in zip(self.embedding_dim, self.transformer_dim)
        ])
        
        
    def forward(self, x, classes):
        classes = [self.weight[l] @ part for l, part in enumerate(classes)]
        x, _ = merge_channel_dim(x)
        classes, _ = merge_channel_dim(classes)
        result = (torch.conj(classes).transpose(-2,-1) @ x.flatten(2)).reshape(x.shape[0], classes.shape[-1], *x.shape[2:])
        
        return result
    
    
#######################################################################################################################
############################################# SE(3) Linear Decoder ####################################################
#######################################################################################################################     

class SE3LinearDecoder(nn.Module):
    def __init__(self, transformer_dim, decoder_dim):
        super(SE3LinearDecoder, self).__init__()
        
        self.transformer_dim = [transformer_dim] if type(transformer_dim) is not list and type(transformer_dim) is not tuple else transformer_dim
        self.decoder_dim = [decoder_dim] if type(decoder_dim) is not list and type(decoder_dim) is not tuple else decoder_dim
        
        assert len(self.transformer_dim) == len(self.decoder_dim)

        self.class_emb = nn.ParameterList([
            nn.Parameter(torch.randn(dim1, dim2, dtype=torch.cfloat)) for dim1, dim2 in zip(self.decoder_dim, self.transformer_dim)])
        
        self.norm = SE3BatchNorm()

    def forward(self, x):
        x_shape = x[0].shape
        x = self.norm(x)
        x = [(self.class_emb[l] @ part.flatten(3)).reshape(x_shape[0], 2*l+1, -1, *x_shape[3:]) 
             for l, part in enumerate(x)]
        
        return x
