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
import torch.nn.functional as F

DROP_ROI_LIST = ["mortor", "car", "fork", "spoon", "knife", "cow", "bus", "cell phone", 
           "carrot", "stop sign", "handbag", "train", "backpack", "suitcase", 
           "scissors", "boat", "orange", "airplane", "apple", "sport ball", "truck", 
           "cat", "tie", "frisbee", "traffic light", "book", "remote", "surfboard", 
           "tennis racket", "dinning table", "airplane", "keyboard", "mouse", 
           "skateboard", "dining table", "sheep", "teddy bear", "zebra", "kite", "bear",'vase','tv']

with open("./yolov8m.yaml", "r") as stream:
    try:
        yolo_yaml = yaml.safe_load(stream)
    except:
        print("Missing yolov8m.yaml in image_process.py")
        raise ValueError("Missing yolov8m.yaml image_process.py")
    
CLASS_MAP = {k:v for k,v in enumerate(yolo_yaml['classes'])}

class MyRoIModel(torch.nn.Module):
  def __init__(self,num_classes):
    super(MyRoIModel, self).__init__()
    self.feature_extractor = torchvision.models.resnet152(weights = ResNet152_Weights.IMAGENET1K_V2)
    self.no_fc = torch.nn.Sequential(*(list(self.feature_extractor.children())[:-1]))
    self.linear = torch.nn.Linear(2048,num_classes)
  def forward(self, input):
    img_features = self.no_fc(input).squeeze(-1).squeeze(-1)
    out = self.linear(img_features)
    return out

class MyImgModel(torch.nn.Module):
  def __init__(self,num_classes):
    super(MyImgModel, self).__init__()
    self.feature_extractor = torchvision.models.resnet152(weights = ResNet152_Weights.IMAGENET1K_V2)
    self.no_fc = torch.nn.Sequential(*(list(self.feature_extractor.children())[:-1]))
    self.linear = torch.nn.Linear(2048,num_classes)
  def forward(self, input):
    img_features = self.no_fc(input).squeeze(-1).squeeze(-1)
    out = self.linear(img_features)
    return out

def convert_img_to_tensor(image):
    
    transforms = v2.Compose([
                        v2.Resize((224,224),antialias=True),  # args.crop_size, by default it is set to be 224
                        v2.ConvertImageDtype(torch.float32),
                        v2.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
                                    ])

    image = transforms(image)

    return image

def load_model(path):
    check_point = torch.load(path,map_location=torch.device('cpu'))
    return check_point

# =============================== ROI PATH ===============================
def merge_boxes(boxes, epsilon):
    # Create a dictionary to store merged boxes for each category
    merged_boxes = {}
    i = 1
    for box in boxes:
        category = box['category']
        coordinates = box['coordinates']
        # Check if there is a merged box for the category
        if category not in merged_boxes:
            merged_boxes[category] = {'coordinates': coordinates, 'count': 1}
        else:
            # Merge coordinates if the box is nearby (based on epsilon)
            current_coords = merged_boxes[category]['coordinates']
            if are_boxes_nearby(current_coords, coordinates, epsilon):
                merged_boxes[category]['coordinates'] = merge_coordinates(current_coords, coordinates)
                merged_boxes[category]['count'] += 1
            else:
                # If not nearby, treat it as a separate box
                merged_boxes[category + "_" + str(i)] = {'coordinates': coordinates, 'count': 1}
            i+=1
    return merged_boxes

def are_boxes_nearby(coords1, coords2, epsilon):
    # Check if the boxes are nearby based on epsilon
    x1a, y1a, x1b, y1b = coords1
    x2a, y2a, x2b, y2b = coords2

    return (
        abs(x1a - x2a) <= epsilon and
        abs(y1a - y2a) <= epsilon and
        abs(x1b - x2b) <= epsilon and
        abs(y1b - y2b) <= epsilon
    )

def merge_coordinates(coords1, coords2):
    # Merge coordinates of two boxes
    x1a, y1a, x1b, y1b = coords1
    x2a, y2a, x2b, y2b = coords2

    merged_xa = min(x1a, x2a)
    merged_ya = min(y1a, y2a)
    merged_xb = max(x1b, x2b)
    merged_yb = max(y1b, y2b)

    return merged_xa, merged_ya, merged_xb, merged_yb

def get_roi(yolo_model, image_path):
    
    results = yolo_model(f'{image_path}',verbose=False)
    box = None

    classes = []
    for r in results:
        box = r.boxes.xyxy
        for c in r.boxes.cls:
            classes.append(CLASS_MAP[c.detach().cpu().numpy().item()])

    boxes = []
    for i,b in enumerate(box):

        x1,y1,x2,y2 = b.detach().cpu().numpy()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        if classes[i] not in DROP_ROI_LIST:
            boxes.append({
                "category":classes[i],
                'coordinates': [x1,y1,x2,y2]
            })
            # print(classes[i], x1,y1,x2,y2)

    return boxes

def roi_predict_wrapper(roi_model, yolo_model, image, eps,list_aspect, image_path, device):

    boxes = get_roi(yolo_model,image_path)

    result = merge_boxes(boxes, eps)
    
    rois_aspect = []
    for category, merged_box in result.items():
        y1, x1, y2, x2 = merged_box['coordinates']
        roi_img = image[:,x1:x2,y1:y2]
        roi_img = convert_img_to_tensor(roi_img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = roi_model(roi_img)
            pred = np.argmax(pred.cpu().numpy(),axis=-1)
            rois_aspect.append(list_aspect[pred[0]])
            
    rois_aspect = list(set(rois_aspect))
    return rois_aspect

def roi_encoder(model,image):
    x = model.conv1(image)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    fc = x.mean(3).mean(2)

    return fc

# =============================== IMAGE PATH ===============================

def image_predict_wrapper(model, list_aspect, image_path ,device):
    image = torchvision.io.read_image(f"{image_path}",mode = torchvision.io.ImageReadMode.RGB)
    img = convert_img_to_tensor(image)
    pred = model(img.unsqueeze(0).to(device))
    pred = torch.sigmoid(pred).squeeze(0)
    pred = pred > 0.6 # in our case, threshold equal 0.45
    pred = pred.cpu().numpy().astype(int)
    pred = np.where(pred==1)[0]
    
    return list(list_aspect[pred])

def image_processing(image_model, roi_model, yolo_model, list_img_path, eps, list_aspect, device):
    
    list_image_aspect = []
    list_roi_aspect = []

    for img_path in list_img_path:
        image = torchvision.io.read_image(f"{img_path}",mode = torchvision.io.ImageReadMode.RGB)
        image_aspect = image_predict_wrapper(image_model, np.asarray(list_aspect),img_path,device)
        roi_aspect = roi_predict_wrapper(roi_model,yolo_model, image, eps,np.asarray(list_aspect), img_path,device)
        try:
            list_image_aspect.extend(image_aspect)
        except:
            pass
        try:
            list_roi_aspect.extend(roi_aspect)
        except:
            pass

    list_image_aspect = set(list_image_aspect)
    list_roi_aspect = set(list_roi_aspect)
    return list_image_aspect, list_roi_aspect

def image_encoder(model,image):
    x = model.conv1(image)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    att = F.adaptive_avg_pool2d(x,[7,7])

    return att

# =============================== CONSTRUCTING AUXILIARY SENTENCE ===============================
def construct_visual_features(yolo_model, list_img_path,eps, NUM_ROI, NUM_IMG, device):
    list_img_features = []
    global_roi_features = [] 
    global_roi_coor = []

    for img_path in list_img_path:
        one_image = torchvision.io.read_image(f"{img_path}",mode = torchvision.io.ImageReadMode.RGB)
        img_transform = convert_img_to_tensor(one_image).unsqueeze(0) 
        list_img_features.append(img_transform)

        list_roi_img = [] 
        list_roi_coor = [] 

        boxes = get_roi(yolo_model, img_path)
        boxes = merge_boxes(boxes, eps)

        if len(boxes) == 0:
            list_roi_img = np.zeros((NUM_ROI,3,224,224))
            
            global_roi_coor.append(np.zeros((NUM_ROI,4)))
            global_roi_features.append(list_roi_img)
            continue
        
        i_roi = 0
        for _, merged_box in boxes.items():
            if i_roi == NUM_ROI:
                break
            y1, x1, y2, x2 = merged_box['coordinates']

            roi_in_image = one_image[:,x1:x2,y1:y2]
            roi_transform = convert_img_to_tensor(roi_in_image).numpy() # 3, 224, 224

            x1, x2, y1, y2 = x1/512, x2/512, y1/512, y2/512
            cv = lambda x: np.clip([x],0.0,1.0)[0]
            x1 = cv(x1)
            x2 = cv(x2)
            y1 = cv(y1)
            y2 = cv(y2) 

            list_roi_coor.append([x1,x2,y1,y2])
            list_roi_img.append(roi_transform)
            i_roi +=1
            
        if i_roi < NUM_ROI:
            for k in range(NUM_ROI - i_roi):
                list_roi_img.append(np.zeros((3,224,224)))
                list_roi_coor.append(np.zeros((4,)))
                
        global_roi_features.append(list_roi_img)
        global_roi_coor.append(list_roi_coor)
        
    t_img_features = torch.zeros((NUM_IMG, 3, 224, 224))
    num_imgs = len(list_img_features)

    if num_imgs >= NUM_IMG:
        for i in range(NUM_IMG):
            t_img_features[i,:] = list_img_features[i]
    else:
        for i in range(NUM_IMG):
            if i < num_imgs:
                t_img_features[i, :] = list_img_features[i]
            else:
                break

    global_roi_features = np.asarray(global_roi_features)
    global_roi_coor = np.asarray(global_roi_coor)

    roi_img_features = np.zeros((NUM_IMG, NUM_ROI, 3, 224, 224))
    roi_coors = np.zeros((NUM_IMG,NUM_ROI,4))

    num_img_roi = len(global_roi_features)

    if num_img_roi >= NUM_IMG:
        for i in range(NUM_IMG):
            roi_img_features[i,:] = global_roi_features[i]
            roi_coors[i,:] = global_roi_coor[i]
    else:
        for i in range(NUM_IMG):
            if i < num_imgs:
                # img_features[(self.max_img_len-num_imgs)+i,:] = imgs[i]
                roi_img_features[i, :] = global_roi_features[i]
                roi_coors[i,:] = global_roi_coor[i,:]
            else:
                break

    roi_img_features = torch.tensor(roi_img_features)
    roi_coors = torch.tensor(roi_coors)

    return t_img_features, roi_img_features, roi_coors
