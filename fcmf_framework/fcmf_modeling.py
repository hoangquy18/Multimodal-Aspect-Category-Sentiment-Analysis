import torch
import torch.nn as nn
import copy
import numpy as np
import math
import torch.nn.functional as F
from transformers import AutoModel
from .mm_modeling import *
from .roi_modeling import *

class FCMF(nn.Module):
    def __init__(self, pretrained_path, num_labels=4, num_imgs = 7, num_roi = 7):
        super(FCMF, self).__init__()
        self.num_labels = num_labels
        self.num_imgs = num_imgs
        self.num_roi = num_roi
        self.bert = FeatureExtractor(pretrained_path)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)
        self.vismap2text = nn.Linear(2048, HIDDEN_SIZE)
        self.roimap2text = nn.Linear(2048, HIDDEN_SIZE)
        self.box_head = BoxMultiHeadedAttention(8,HIDDEN_SIZE)

        self.text2img_attention = BertCrossEncoder()
        self.text2img_pooler = BertPooler()
        self.text2roi_pooler = BertPooler()

        self.mm_attention = MultimodalEncoder()
        self.text_pooler = BertPooler()
        self.classifier = nn.Linear(HIDDEN_SIZE, num_labels)

    def forward(self, input_ids, visual_embeds_att, roi_embeds_att, roi_coors = None, token_type_ids=None, attention_mask=None, added_attention_mask=None):

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

        seq_len = sequence_output.size()[1]

        # visual size: batch_size, num_img, 49,2048 
        # roi size: batch_size, num_img, num_roi, 49,2048
        list_h_i = []
        list_r_i = []
        for i in range(self.num_imgs):
            # Image-guided Attention
            ## PROCESS LIST OF IMAGES

            one_img_embeds = visual_embeds_att[:,i,:] # each image index: batch_size, 49, 2048
            converted_img_embed_map = self.vismap2text(one_img_embeds)  # self.batch_size, 49, hidden_dim

            img_mask = added_attention_mask[:,:49]
            extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
            extended_img_mask = extended_img_mask.to(dtype=converted_img_embed_map.dtype) # fp16 compatibility
            extended_img_mask = (1.0 - extended_img_mask) * -10000.0

            text2img_cross_attention = self.text2img_attention(sequence_output, converted_img_embed_map, extended_img_mask)
            text2img_output_layer = text2img_cross_attention[-1]
            text2img_cross_output = self.text2img_pooler(text2img_output_layer) # self.batch_size, 768
            transpose_text2img_embed = text2img_cross_output.unsqueeze(1) # self.batch_size, 1, 768

            list_h_i.append(transpose_text2img_embed) 

            # Geometric Roi-aware 
            ### EACH IMAGES HAVE n ROI
            text2roi_mask = added_attention_mask[:,:seq_len + self.num_roi]
            text2roi_mask = text2roi_mask.unsqueeze(1).unsqueeze(2)
            text2roi_mask = text2roi_mask.to(dtype=text2roi_mask.dtype)  # fp16 compatibility
            text2roi_mask = (1.0 - text2roi_mask) * -10000.0

            roi_at_i_img = roi_embeds_att[:,i,:] # batch_size, num_roi, 2048
            converted_roi_embed_map = self.roimap2text(roi_at_i_img) # batch_size, num_roi, hidden_dim
            
            # roi_coor: batch_size, num_img, num_roi , 4 
            relative_roi = self.box_head(converted_roi_embed_map,converted_roi_embed_map,converted_roi_embed_map,roi_coors[:,i,:]) # batch_size, num_roi, hidden_dim

            text_roi_output = torch.cat((sequence_output,relative_roi), dim=1) # batch_size, seq_len + num_roi, hidden_dim

            roi_multimodal_encoder = self.mm_attention(text_roi_output, text2roi_mask)
            roi_att_text_output_layer = roi_multimodal_encoder[-1] # batch_size, seq_len + num_roi, hidden_dim
            roi_pooling = self.text2roi_pooler(roi_att_text_output_layer)
            transpose_roi_embed = roi_pooling.unsqueeze(1) # self.batch_size, 1, 768

            list_r_i.append(transpose_roi_embed) 

        all_h_i_features = torch.cat(list_h_i, dim = 1) # batch_size, num_img, 768
        all_r_i_features = torch.cat(list_r_i, dim = 1) # batch_size, num_img*num_roi, 768

        fusion = torch.cat((sequence_output[:,0,:].unsqueeze(1), all_h_i_features,all_r_i_features),dim=1) # batch_size, 1+num_img+num_img*num_roi, 768
    
        comb_attention_mask = added_attention_mask[:,:self.num_imgs + self.num_imgs + 1]  
        extended_attention_mask = comb_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=extended_attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        final_multimodal_encoder = self.mm_attention(fusion, extended_attention_mask)
        final_multimodal_encoder = final_multimodal_encoder[-1]
        
        text_output = self.text_pooler(final_multimodal_encoder)
        pooled_output = self.dropout(text_output)

        logits = self.classifier(pooled_output)

        return logits   