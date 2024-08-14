import torch 
import torch.nn as nn
from math import sqrt
from SteerableSegmenter2D.utils import get_pos_encod
from SteerableSegmenter2D.conv_layers import HNonLinearity2D, SteerableBatchNorm2D
        
#######################################################################################################################
############################################## Multihead Self Attention ###############################################
#######################################################################################################################

class SE2MultiSelfAttention(nn.Module):
    def __init__(self, transformer_dim, n_head, max_m, add_pos_enc = True):
        super(SE2MultiSelfAttention, self).__init__()

        # Layer Design
        if not transformer_dim % n_head == 0 :
            raise ValueError(f"Transformer dimension ({transformer_dim}) is not divisible by number of heads ({n_head}).")
        self.query_dim = transformer_dim // n_head
        self.n_head = n_head
        self.max_m = max_m
        self.add_pos_enc = add_pos_enc

        # Layer Parameters
        self.embeddings_real = nn.Parameter(torch.randn(3, 1, max_m, n_head * self.query_dim, transformer_dim, dtype = torch.float))
        self.encoding_real = nn.Parameter(torch.randn(self.max_m, n_head, self.query_dim, dtype = torch.float))
        self.soft_real =  nn.Parameter(torch.randn(self.max_m, n_head, dtype = torch.float))
        self.out_real = nn.Parameter(torch.randn(max_m, transformer_dim, n_head * self.query_dim, dtype = torch.float))

        self.embeddings_imag = nn.Parameter(torch.randn(3, 1, max_m, n_head * self.query_dim, transformer_dim, dtype = torch.float))
        self.encoding_imag = nn.Parameter(torch.randn(self.max_m, n_head, self.query_dim, dtype = torch.float))
        self.soft_imag =  nn.Parameter(torch.randn(self.max_m, n_head, dtype = torch.float))
        self.out_imag = nn.Parameter(torch.randn(max_m, transformer_dim, n_head * self.query_dim, dtype = torch.float))

        self.pos_enc = None

    def forward(self, x):
        self.embeddings = torch.complex(self.embeddings_real, self.embeddings_imag)
        self.soft = torch.complex(self.soft_real, self.soft_imag)
        self.encoding = torch.complex(self.encoding_real, self.encoding_imag)
        self.out = torch.complex(self.out_real, self.out_imag)
        
        # Query, Key and Value Embeddings
        E = (self.embeddings @ x).reshape(3, x.shape[0], self.max_m, self.n_head, self.query_dim, -1)
        Q, K, V = torch.conj(E[0]), E[1], E[2]
        
        # Attention Weights
        A = torch.einsum('bkhqN, bkhqM, kh -> bhNM', K, Q, self.soft)
        
        # Positional Encoding
        if self.add_pos_enc:
            if self.pos_enc ==  None or not self.pos_enc.shape[-1] == x.shape[-1] or not self.pos_enc.shape[0] == x.shape[1]:
                self.pos_enc = get_pos_encod(x.shape[-1], self.max_m).to(A.device)
            pos = torch.einsum('bkhqM, kMN, kh, khq -> bhMN', Q, self.pos_enc, self.soft, self.encoding)
            A = A + pos
        A = nn.functional.softmax(A.abs() / sqrt(self.query_dim), dim = -2)

        # Output
        result = self.out @ (torch.einsum('bkhqN, bhNM -> bkhqM', V, A.type(torch.cfloat))).flatten(2,3)
        result = result.reshape(x.shape[0], self.max_m, -1, *x.shape[3:])


        return result

        
    
#######################################################################################################################
############################################## Positionwise Feedforward ###############################################
#######################################################################################################################    

class PositionwiseFeedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_m):
        super(PositionwiseFeedforward, self).__init__()

        self.max_m = max_m

        self.weights1_real = nn.Parameter(torch.randn(max_m, hidden_dim, input_dim, dtype = torch.float))
        self.weights1_imag = nn.Parameter(torch.randn(max_m, hidden_dim, input_dim, dtype = torch.float))
        self.weights2_real = nn.Parameter(torch.randn(max_m, input_dim, hidden_dim, dtype = torch.float))
        self.weights2_imag = nn.Parameter(torch.randn(max_m, input_dim, hidden_dim, dtype = torch.float))

        self.eps = 1e-5
        self.nonlinearity = HNonLinearity2D(hidden_dim, max_m)

    def forward(self, x):
        self.weights1 = torch.complex(self.weights1_real, self.weights1_imag)
        self.weights2 = torch.complex(self.weights2_real, self.weights2_imag)

        x = self.weights1 @ x
        x = self.nonlinearity(x)
        x = self.weights2 @ x
        return x   

#######################################################################################################################
################################################# SE(2) Transformer ###################################################
####################################################################################################################### 

class SE2Transformer(nn.Module):
    def __init__(self, transformer_dim, n_head, hidden_dim, max_m, add_pos_enc = True):
        super(SE2Transformer, self).__init__()

        # Layer Design
        self.multihead_attention = SE2MultiSelfAttention(transformer_dim, n_head, max_m, add_pos_enc)
        self.positionwise_feedforward = PositionwiseFeedforward(transformer_dim, hidden_dim, max_m)

        self.layer_norm1 = SteerableBatchNorm2D()
        self.layer_norm2 = SteerableBatchNorm2D()

    def forward(self, x):
        x = self.multihead_attention(self.layer_norm1(x)) + x
        x = self.positionwise_feedforward(self.layer_norm2(x)) + x
        
        return x
    
#######################################################################################################################
############################################## SE(2) Transformer Encoder ##############################################
####################################################################################################################### 

class SE2TransformerEncoder(nn.Module):
    def __init__(self, transformer_dim, n_head, max_m, n_layers = 1, add_pos_enc = True):
        super(SE2TransformerEncoder, self).__init__()

        # Layer Design
        self.transformer_encoder = torch.nn.Sequential(
            *[SE2Transformer(transformer_dim, n_head, 2*transformer_dim, max_m, add_pos_enc) for _ in range(n_layers)]
        )

    def forward(self, x):
        x_shape = x.shape
        x = x.flatten(3) # shape : batch x max_m x channel x N
        x = self.transformer_encoder(x)
        x = x.reshape(*x_shape)
        
        return x
    
#######################################################################################################################
############################################## SE(2) Transformer Decoder ##############################################
####################################################################################################################### 

class SE2TransformerDecoder(nn.Module):
    def __init__(self, transformer_dim, n_head, max_m, n_classes, n_layers, add_pos_enc = True):
        super(SE2TransformerDecoder, self).__init__()

        self.transformer_dim = transformer_dim
        self.scale = transformer_dim ** -0.5
        self.n_classes = n_classes
        self.max_m = max_m
        
        self.transformer_encoder = torch.nn.Sequential(
            *[SE2Transformer(transformer_dim, n_head, 2*transformer_dim, max_m, add_pos_enc) for _ in range(n_layers)]
        )

        self.class_embed_real = nn.Parameter(torch.randn(1, 1, transformer_dim, n_classes, dtype=torch.float))
        self.class_embed_imag = nn.Parameter(torch.randn(1, 1, transformer_dim, n_classes, dtype=torch.float))

        self.norm = SteerableBatchNorm2D()
        self.C = torch.tensor([[[(m1+m2-m)%max_m == 0 for m2 in range(max_m)]
                           for m1 in range(max_m)] for m in range(max_m)]).type(torch.cfloat)

    def forward(self, x):
        x_shape = x.shape
        class_embed = torch.complex(self.class_embed_real, self.class_embed_imag)
        pad = torch.zeros(x_shape[0], self.max_m-1, self.transformer_dim, self.n_classes, dtype=torch.cfloat, device=class_embed.device)
        class_embed = torch.cat((class_embed.expand(x_shape[0], 1, -1, -1), pad), dim=1)

        x = torch.cat((x.flatten(3), class_embed), -1)
        x = self.norm(self.transformer_encoder(x))
        patches, cls_seg_feat = x[..., : -self.n_classes], x[..., -self.n_classes :]
        patches = patches.reshape(x_shape[0], self.max_m, -1, *x_shape[3:])
        
  
        return patches, cls_seg_feat
    
class SE2ClassEmbeddings(nn.Module):
    def __init__(self, transformer_dim, embedding_dim, max_m):
        super(SE2ClassEmbeddings, self).__init__()
        
        
        self.weight_real = nn.Parameter(torch.randn(embedding_dim, transformer_dim, dtype=torch.float))
        self.weight_imag = nn.Parameter(torch.randn(embedding_dim, transformer_dim, dtype=torch.float))
        self.C = torch.tensor([[[(m1+m2-m)%max_m == 0 for m2 in range(max_m)] 
                           for m1 in range(max_m)] for m in range(max_m)]).type(torch.cfloat)
        
    def forward(self, x, classes):
        C = self.C.to(classes.device)
        self.weight = torch.complex(self.weight_real, self.weight_imag)
        classes = self.weight @ classes
        x = torch.einsum('lmn, bmeXY, bneC -> blCXY', C, x, classes)
        
        return x
    
    
#######################################################################################################################
############################################# SE(2) Linear Decoder ####################################################
#######################################################################################################################     

class SE2LinearDecoder(nn.Module):
    def __init__(self, transformer_dim, decoder_dim, max_m):
        super(SE2LinearDecoder, self).__init__()

        self.max_m = max_m
        self.decoder_dim = decoder_dim
        self.transformer_dim = transformer_dim

        self.class_emb_real = nn.Parameter(torch.randn(max_m, decoder_dim, transformer_dim, dtype=torch.float))
        self.class_emb_imag = nn.Parameter(torch.randn(max_m, decoder_dim, transformer_dim, dtype=torch.float))
        self.norm = SteerableBatchNorm2D()

    def forward(self, x):
        self.class_emb = torch.complex(self.class_emb_real, self.class_emb_imag)
        x_shape = x.shape
        x = self.norm(x)
        x = self.class_emb @ x.flatten(3)
        x = x.reshape(x.shape[0], self.max_m, -1, *x_shape[3:])
        
        return x