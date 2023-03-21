from torch import nn as nn  
from .ActivationFunc import activation_getter 

class ResidualBlock(nn.Module):
    def __init__(self, num_features, activation, dropout=False): 
        
        super().__init__() 
        
        self.dropout = dropout 
        self.activation = activation_getter(activation) 

        # Residual Block nn layers 
        self.lin1 = nn.Linear(num_features, num_features) 
        nn.init.xavier_uniform_(self.lin1.weight.data) 
        self.lin1.bias.data.zero_() 

        self.lin2 = nn.Linear(num_features, num_features, bias=True) 
        nn.init.xavier_uniform_(self.lin2.weight.data) 
        self.lin2.bias.data.zero_() 

        if dropout:
            self.dropout_layer = nn.Dropout()

    def forward(self, x):
       
        x_res = x 
       
        x = self.activation(x) 
        x = self.lin1(x) 
        if self.dropout:
            x = self.dropout_layer(x) 
        x = self.lin2(x) 
        if self.dropout:
            x = self.dropout_layer(x) 
        
        return x + x_res 


class EmbeddingLayer(nn.Module):
    
    def __init__(self, num_embeddings, out_features):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, out_features) 
        self.embedding.weight.data.uniform_(-1.732, 1.732) 

    def forward(self, Z):
        """
        :param Z, one-dimensional feature 
        :output v_i, embeding of  
        """
        v_i = self.embedding(Z) 
        return v_i 


def NormLayer(num_features, mode="batch"):
    if mode == "batch":
        return nn.BatchNorm1d(num_features, momentum=0.4, affine=True) 
    elif mode == "layer":
        return nn.LayerNorm(num_features, elementwise_affine=True) 
    elif mode == "node":
        return nn.LayerNorm(num_features, elementwise_affine=False) 
    elif mode == "none":
        return nn.Identity() 
    else:
        raise NotImplemented 
    
