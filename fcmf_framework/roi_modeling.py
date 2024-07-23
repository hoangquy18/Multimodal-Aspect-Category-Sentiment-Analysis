import torch
import torch.nn as nn
import copy
import numpy as np
import math
import torch.nn.functional as F
from transformers import AutoModel


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
    '''

    N = value.size()[:2]
    dim_k = key.size(-1)
    dim_g = box_relation_embds_matrix.size()[-1]

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value

    #attention weights
    scaled_dot = torch.matmul(w_q,w_k)
    scaled_dot = scaled_dot / np.sqrt(dim_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)

    #w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix
    w_a = scaled_dot
    #w_a = scaled_dot.view(N,N)

    # multiplying log of geometric weights by feature weights
    w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)

    output = torch.matmul(w_mn,w_v)

    return output, w_mn

class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, trignometric_embedding=True, legacy_extra_skip=False, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.trignometric_embedding=trignometric_embedding
        self.legacy_extra_skip = legacy_extra_skip

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        if self.trignometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        #matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True),8)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def BoxRelationalEmbedding(self,f_g, dim_g=64, wave_len=1000, trignometric_embedding= True):
        """
        Given a tensor with bbox coordinates for detected objects on each batch image,
        this function computes a matrix for each image

        with entry (i,j) given by a vector representation of the
        displacement between the coordinates of bbox_i, and bbox_j

        input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
        output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
        """
        #returns a relational embedding for each pair of bboxes, with dimension = dim_g
        #follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

        batch_size = f_g.size(0)

        x_min, x_max, y_min, y_max = torch.chunk(f_g, 4, dim=-1)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.

        #cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
        delta_x = cx - cx.view(batch_size, 1, -1)
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(batch_size, 1, -1)
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(batch_size, 1, -1))
        delta_h = torch.log(h / h.view(batch_size, 1, -1))

        matrix_size = delta_h.size()
        delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
        delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
        delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
        delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

        position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

        if trignometric_embedding == True:
            feat_range = torch.arange(dim_g / 8).to('cuda')
            dim_mat = feat_range / (dim_g / 8)
            dim_mat = 1. / (torch.pow(wave_len, dim_mat))

            dim_mat = dim_mat.view(1, 1, 1, -1)
            position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
            position_mat = 100. * position_mat

            mul_mat = position_mat * dim_mat
            mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
            sin_mat = torch.sin(mul_mat)
            cos_mat = torch.cos(mul_mat)
            embedding = torch.cat((sin_mat, cos_mat), -1)
        else:
            embedding = position_mat
        return (embedding)

    def forward(self, input_query, input_key, input_value, input_box, mask=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = input_query.size(0)

        #tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        relative_geometry_embeddings = self.BoxRelationalEmbedding(input_box, trignometric_embedding= self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1,self.dim_g)
        flatten_relative_geometry_embeddings = flatten_relative_geometry_embeddings.to(dtype=input_query.dtype) # FP16
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)

        # self.WGs = self.WGs.to(dtype=next(self.parameters()).dtype)
        
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head),1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        # An extra internal skip connection is added. This is only
        # kept here for compatibility with some legacy models. In
        # general, there is no advantage in using it, as there is
        # already an outer skip connection surrounding this layer.
        if self.legacy_extra_skip:
            x = input_value + x

        return self.linears[-1](x)
