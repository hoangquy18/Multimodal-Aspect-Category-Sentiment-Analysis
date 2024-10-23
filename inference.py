from torchvision.models import resnet152,ResNet152_Weights,resnet50, ResNet50_Weights
import torch
import torchvision
import numpy as np
from torchvision.transforms import v2
from ultralytics import YOLO
import yaml
import cv2 
from transformers import AutoTokenizer
from underthesea import word_tokenize,text_normalize
import torch.nn as nn
import copy
import numpy as np
import math
import torch.nn.functional as F
from transformers import AutoModel
from fcmf_framework.fcmf_modeling import FCMF
from text_preprocess import *
from fcmf_framework.image_process import *
import argparse
from loguru import logger
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']
IMG_ASPECT = ['Food', 'Room', 'Facilities', 'Service', 'Public_area']
POLARITY = ['None','Negative','Neutral','Positive']

YOLO_PATH = './yolov8m.pt'
WEIGHT_ROI_PATH = './weight_roi_resnet152.pth'
WEIGHT_IMAGE_PATH = './weight_image_resnet152.pth'
FCMF_CHECKPOINT = './4_relative_mfam_model.pth'
VISUAL_MODEL_CHECKPOINT = './4_visual_model.pth'

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
logger.add("file_{time}.log")

# =============================== LOAD ROI MODEL ===============================
def load_yolo_roi_model(yolo_path,weight_roi_path):
    roi_model = MyRoIModel(len(IMG_ASPECT)) # No Location
    roi_model = roi_model.to(device)

    yolo_model = YOLO(yolo_path)
    yolo_model.to(device)

    try:
        checkpoint = load_model(weight_roi_path)
    except:
        logger.error("Wrong RoI weight path!!!")
        raise ValueError("Wrong RoI weight path!!!")
    
    roi_model.load_state_dict(checkpoint['model_state_dict'])

    roi_model.eval()

    return yolo_model, roi_model

# =============================== LOAD IMAGE MODEL ===============================
def load_image_model(weight_image_path):
    image_model = MyImgModel(len(IMG_ASPECT)) # No Location
    image_model = image_model.to(device)

    try:
        checkpoint = load_model(weight_image_path)
    except:
        logger.error("Wrong Image weight path!!!")
        raise ValueError("Wrong Image weight path!!!")

    image_model.load_state_dict(checkpoint['model_state_dict'])

    image_model.eval()

    return image_model

# ============================  LOADING TRAINED MODEL ============================ 
def load_fcmf_model(fcmf_checkpoint,visual_checkpoint, pretrained_model, num_imgs, num_rois):
    fcmf_model = FCMF(pretrained_model,num_imgs=num_imgs,num_roi=num_rois)
    fcmf_model.to(device)

    visual_model = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2)
    visual_model.to(device)
    try:
        fcmf_checkpoint = load_model(fcmf_checkpoint)
    except:
        logger.error("Wrong FCMF weight path!!!")
        raise ValueError("Wrong FCMF weight path!!!")
    
    fcmf_model.load_state_dict(fcmf_checkpoint['model_state_dict'])
    
    try:
        visual_checkpoint = load_model(visual_checkpoint)
    except:
        logger.error("Wrong visual weight path!!!")
        raise ValueError("Wrong visual weight path!!!")
    
    visual_model.load_state_dict(visual_checkpoint['model_state_dict'])
    
    return fcmf_model, visual_model

# ============================  GETTING VISUAL FEATURES ============================ 
def get_visual_features(yolo_model, visual_model, list_image_path, num_imgs, num_rois, device):
    t_img_features, roi_img_features, roi_coors = construct_visual_features(yolo_model,list_image_path, 30, num_rois, num_imgs, device)
    t_img_features = t_img_features.unsqueeze(0)
    t_img_features = t_img_features.float().to(device)
    roi_img_features = roi_img_features.unsqueeze(0)
    roi_img_features = roi_img_features.float().to(device)
    roi_coors = roi_coors.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_img = []
        encoded_roi = []

        for img_idx in range(num_imgs):
            img_features = image_encoder(visual_model,t_img_features[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1) # batch_size, 49, 2048
            encoded_img.append(img_features)

            roi_f = []
            for roi_idx in range(num_rois):
                roi_features = roi_encoder(visual_model,roi_img_features[:,img_idx,roi_idx,:]).squeeze(1) # batch_size, 1, 2048
                roi_f.append(roi_features)

            roi_f = torch.stack(roi_f,dim=1)
            encoded_roi.append(roi_f)

        encoded_img = torch.stack(encoded_img,dim=1) # batch_size, num_img, 49, 2048   
        encoded_roi = torch.stack(encoded_roi,dim=1) # batch_size, num_img, num_roi, 49,2048

    return roi_coors, encoded_img, encoded_roi

# ============================  FCMF PREDICTION ============================ 
def fcmf_predict_wrapper(tokenizer, text, IMG_ASPECT, ASPECT, list_image_path, num_imgs, num_rois, device):
    # ====== ASPECT PREDICTION ======
    print("============ LOADING MODEL ============")
    logger.info("loading model")
    yolo_model, roi_model = load_yolo_roi_model(YOLO_PATH, WEIGHT_ROI_PATH)
    image_model = load_image_model(WEIGHT_IMAGE_PATH)
    fcmf_model, visual_model = load_fcmf_model(FCMF_CHECKPOINT, VISUAL_MODEL_CHECKPOINT,pretrained_model,num_imgs, num_rois)

    logger.info('construct features')
    print("============ CONSTRUCT FEATURES ============")
    list_image_aspect, list_roi_aspect = image_processing(image_model,roi_model, yolo_model, list_image_path, 30, IMG_ASPECT, device)
    joined_aspect = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}" # RIGHT PATH FOR AUXILIARY SENTENCE
    joined_aspect = joined_aspect.lower().replace('_',' ')

    # VISUAL FEATURES
    roi_coors, encoded_img, encoded_roi = get_visual_features(yolo_model, visual_model, list_image_path,num_imgs, num_rois, device)
    
    logger.info('making prediction')
    print("============ MAKING PREDICTION ============")
    rs = {asp:'None' for asp in ASPECT}
    for id_asp in range(len(ASPECT)):
        asp = ASPECT[id_asp]
        combine_text = f"{asp} </s></s> {text}"
        combine_text = combine_text.lower().replace('_',' ')
        tokens = tokenizer(combine_text, joined_aspect, max_length=170,truncation='only_first',padding='max_length', return_token_type_ids=True)

        input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(tokens['token_type_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0).to(device)
        added_input_mask =torch.tensor( [1] * (170+49)).unsqueeze(0).to(device)

        logits = fcmf_model(
            input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, 
            added_attention_mask = added_input_mask,
            visual_embeds_att = encoded_img,
            roi_embeds_att = encoded_roi,
            roi_coors = roi_coors
        )
        pred = np.argmax(logits.detach().cpu(),axis = -1)

        rs[ASPECT[id_asp]] = POLARITY[pred[0]]

    logger.success("Done")
    logger.success(f'{rs}')
    return rs
# ================== TEXT ==================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",
                    type=str,
                    required=True,
                    help="text input")

    parser.add_argument("--image_list", 
                        "--names-list", 
                        nargs='+', help = "Optional. List of image associated with text.")

    parser.add_argument("--num_images",
                        type=int,
                        default = 7,
                        required=False,
                        help="number of images")

    parser.add_argument("--num_rois",
                        type=int,
                        default = 4,
                        required=False,
                        help="number of RoIs")

    parser.add_argument("--pretrained_model",
                        type=str,
                        required=False,
                        default='xlm-roberta-base',
                        help="pretrained model for FCMF framework")

    args = parser.parse_args()

    num_rois = args.num_rois
    num_imgs = args.num_imgs
    list_image_path = args.image_list

    text = args.text
    normalize_class = TextNormalize()
    text = normalize_class.normalize(text_normalize(convert_unicode(text)))    

    pretrained_model = args.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    print(f"Using {num_imgs} images and {num_rois} RoIs.")
    logger.info(f"Using {num_imgs} images and {num_rois} RoIs.")
    print(f"Using {pretrained_model} for text features extraction.")
    logger.info(f"Using {pretrained_model} for text features extraction.")

    result = fcmf_predict_wrapper(
                tokenizer = tokenizer,\
                text = text, \
                IMG_ASPECT = IMG_ASPECT, \
                ASPECT = ASPECT, \
                list_image_path = list_image_path, \
                num_imgs = num_imgs, \
                num_rois = num_rois, \
                device = device
            )
    print(result)
    