import torch 
import torch.nn as nn
from math import sqrt
from Steerable.nn.Steerable2d.utils import get_pos_encod
from Steerable.nn.Steerable2d.conv_layers import SE2NormNonLinearity, SE2BatchNorm, ComplexDropout
        
#######################################################################################################################
############################################## Multihead Self Attention ###############################################
#######################################################################################################################

class SE2MultiSelfAttention(nn.Module):
    def __init__(self, transformer_dim, n_head, max_m, dropout = 0.1, add_pos_enc = True):
        super(SE2MultiSelfAttention, self).__init__()

        # Layer Design
        if not transformer_dim % n_head == 0 :
            raise ValueError(f"Transformer dimension ({transformer_dim}) is not divisible by number of heads ({n_head}).")
        self.query_dim = transformer_dim // n_head
        self.n_head = n_head
        self.max_m = max_m
        self.add_pos_enc = add_pos_enc
        self.dropout = ComplexDropout(p = dropout)

        # Layer Parameters
        self.embeddings = nn.Parameter(torch.randn(3, 1, max_m, n_head, self.query_dim, transformer_dim, dtype = torch.cfloat))
        self.encoding = nn.Parameter(torch.randn(2, self.max_m, n_head, 1, self.query_dim, 1, dtype = torch.cfloat))
        self.out = nn.Parameter(torch.randn(max_m, transformer_dim, n_head * self.query_dim, dtype = torch.cfloat))

        self.pos_enc = None

    def forward(self, x):
        x_shape = x.shape
        x = x.flatten(3) # shape : batch x max_m x channel x N
        
        # Query, Key and Value Embeddings
        E = (self.embeddings @ x.unsqueeze(2))
        Q, K, V = torch.conj(E[0].transpose(-2,-1)), E[1], E[2]
        
        # Scores
        A = Q @ K
                
        # Positional Encoding
        if self.add_pos_enc:
            if self.pos_enc ==  None:
                self.pos_enc = get_pos_encod(x_shape[-2:], self.max_m).to(Q.device)
                
            pos = (self.encoding * self.pos_enc)
            A = A + (Q.unsqueeze(-2) @ pos[0]).squeeze(-2)
        
        # Attention Weights
        A = torch.sum(A, dim=1, keepdim=True)
        A = nn.functional.softmax(A.abs() / sqrt(self.query_dim), dim = -1).type(torch.cfloat)
        A = self.dropout(A)
 
        # Output
        result = V @ A.transpose(-2,-1)
        if self.add_pos_enc:
            result = result + (pos[1] @ A.unsqueeze(-1)).squeeze(-1).transpose(-2,-1)
        
        # Mixing Heads
        result = self.out @ result.flatten(2,3)
        result = result.reshape(*x_shape)

        return result

#######################################################################################################################
############################################## Positionwise Feedforward ###############################################
#######################################################################################################################    

class SE2PositionwiseFeedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_m, dropout = 0.1):
        super(SE2PositionwiseFeedforward, self).__init__()

        self.max_m = max_m

        self.weights1 = nn.Parameter(torch.randn(max_m, hidden_dim, input_dim, dtype = torch.cfloat))
        self.weights2 = nn.Parameter(torch.randn(max_m, input_dim, hidden_dim, dtype = torch.cfloat))
        self.dropout1 = ComplexDropout(p = dropout)
        self.dropout2 = ComplexDropout(p = dropout)
        
        self.eps = 1e-5
        self.nonlinearity = SE2NormNonLinearity(hidden_dim, max_m, nonlinearity=nn.GELU())

    def forward(self, x):
        x_shape = x.shape
        x = x.flatten(3) # shape : batch x max_m x channel x N

        x = self.weights1 @ x
        x = self.nonlinearity(x)
        x = self.dropout1(x)
        x = self.weights2 @ x
        x = self.dropout2(x)
        
        x = x.reshape(*x_shape)
        return x   

#######################################################################################################################
################################################# SE(2) Transformer ###################################################
####################################################################################################################### 

class SE2Transformer(nn.Module):
    def __init__(self, transformer_dim, n_head, hidden_dim, max_m, dropout = 0.1, add_pos_enc = True):
        super(SE2Transformer, self).__init__()

        # Layer Design
        self.multihead_attention = SE2MultiSelfAttention(transformer_dim, n_head, max_m, dropout, add_pos_enc)
        self.positionwise_feedforward = SE2PositionwiseFeedforward(transformer_dim, hidden_dim, max_m, dropout)

        self.layer_norm1 = SE2BatchNorm()
        self.layer_norm2 = SE2BatchNorm()

    def forward(self, x):
        x = self.multihead_attention(self.layer_norm1(x)) + x
        x = self.positionwise_feedforward(self.layer_norm2(x)) + x
        return x
    
#######################################################################################################################
############################################## SE(2) Transformer Encoder ##############################################
####################################################################################################################### 

class SE2TransformerEncoder(nn.Module):
    def __init__(self, transformer_dim, n_head, max_m, n_layers = 1, dropout = 0.1, add_pos_enc = True):
        super(SE2TransformerEncoder, self).__init__()

        # Layer Design
        self.transformer_encoder = torch.nn.Sequential(
            *[SE2Transformer(transformer_dim, n_head, 2*transformer_dim, max_m, dropout, add_pos_enc) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x
    
#######################################################################################################################
############################################## SE(2) Transformer Decoder ##############################################
####################################################################################################################### 

class SE2TransformerDecoder(nn.Module):
    def __init__(self, transformer_dim, n_head, max_m, n_classes, n_layers, dropout = 0.1, add_pos_enc = True):
        super(SE2TransformerDecoder, self).__init__()

        self.transformer_dim = transformer_dim
        self.scale = transformer_dim ** -0.5
        self.n_classes = n_classes
        self.max_m = max_m
        
        self.transformer_encoder = torch.nn.Sequential(
            *[SE2Transformer(transformer_dim, n_head, 2*transformer_dim, max_m, dropout, add_pos_enc) for _ in range(n_layers)]
        )

        self.class_embed = nn.Parameter(torch.randn(1, 1, transformer_dim, n_classes, dtype=torch.cfloat))

        self.norm = SE2BatchNorm()
        self.C = torch.tensor([[[(m1+m2-m)%max_m == 0 for m2 in range(max_m)]
                           for m1 in range(max_m)] for m in range(max_m)]).type(torch.cfloat)

        self.pos_enc = None
    def forward(self, x):
        if self.pos_enc == None:
            pos = get_pos_encod(x.shape[-2:], self.max_m).to(x.device)
            self.pos_enc = torch.zeros(self.max_m, 1, pos.shape[2]+self.n_classes, 1, pos.shape[4]+self.n_classes, dtype=torch.cfloat, device=x.device)
            self.pos_enc[:,:,:pos.shape[2],:,:pos.shape[4]] = pos
            for module in self.transformer_encoder:
                module.multihead_attention.pos_enc = self.pos_enc
        
        x_shape = x.shape
        pad = torch.zeros(x_shape[0], self.max_m-1, self.transformer_dim, self.n_classes, 
                          dtype=torch.cfloat, device=self.class_embed.device)
        class_embed = torch.cat((self.class_embed.expand(x_shape[0], 1, -1, -1), pad), dim=1)

        x = torch.cat((x.flatten(3), class_embed), -1)
        x = self.norm(self.transformer_encoder(x))
        
        patches, cls_seg_feat = x[..., : -self.n_classes], x[..., -self.n_classes :]
        patches = patches.reshape(x_shape[0], self.max_m, -1, *x_shape[3:])
        
  
        return patches, cls_seg_feat

class SE2ClassEmbeddings(nn.Module):
    def __init__(self, transformer_dim, embedding_dim, max_m):
        super(SE2ClassEmbeddings, self).__init__()
        self.weight = nn.Parameter(torch.randn(max_m, embedding_dim, transformer_dim, dtype=torch.cfloat))
        
    def forward(self, x, classes):
        classes = (self.weight @ classes).flatten(1,2)
        result = torch.conj(classes).transpose(-2,-1) @ x.flatten(1,2).flatten(2)
        result = result.reshape(x.shape[0], classes.shape[-1], *x.shape[-2:])
        #x = torch.einsum('bmeXY, bmeC -> bCXY', x, torch.conj(classes))
        
        return result
   
