import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModel
import re
import torch
import os
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import argparse
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm,trange
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,confusion_matrix
from torchvision.models import resnet152,ResNet152_Weights,resnet50, ResNet50_Weights
import random
import logging

logging.basicConfig(filename='roi_categories.log',format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class RoiDataset(Dataset):
    def __init__(self, data, root_dir, ASPECT):
        self.image_label = data
        self.root_dir = root_dir
        self.ASPECT = ASPECT
    def __len__(self):
        return self.image_label.shape[0]
    def __getitem__(self,index):
        image_name = self.image_label.loc[index, "file_name"] + ".png"
        x1, x2, y1, y2 = self.image_label.iloc[index,1:4+1].values

        image = torchvision.io.read_image(os.path.join(self.root_dir,image_name),mode = torchvision.io.ImageReadMode.RGB) # 3, 512, 512
        image = image[:,x1:x2,y1:y2]

        text_lb = self.image_label.loc[index,'label']
        num_lb = self.ASPECT.index(text_lb)        

        transforms = v2.Compose([
                            v2.Resize((224,224),antialias=True),  # args.crop_size, by default it is set to be 224
                            v2.RandomHorizontalFlip(),
                            v2.ConvertImageDtype(torch.float32),
                            v2.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))
                                        ])

        image = transforms(image)

        return {
            "image": image,
            "label": num_lb
        }
        
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

def load_model(path):
    check_point = torch.load(path,map_location=torch.device('cpu'))
    return check_point

def save_model(path, model, epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    },path)

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, _ \
      = precision_recall_fscore_support(y_true, y_pred,zero_division=0.0,labels = [0,1,2,3,4])
    return p_macro, r_macro, f_macro

def convert_img_to_tensor(img):

    transforms = v2.Compose([
                        v2.Resize((224,224),antialias=True),  # args.crop_size, by default it is set to be 224
                        # v2.RandomHorizontalFlip(),
                        v2.ConvertImageDtype(torch.float32),
                        v2.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
                                    ])

    image = transforms(img)

    return image
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",
                        default='../image',
                        type=str,
                        required=True,
                        help="The image folder dir. Should contain the list of images files for the MACSA task.")
    parser.add_argument("--roi_label_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The labeled RoI. Should contain the list of annotated RoI files for the MACSA task.")

    parser.add_argument("--weight_path",
                    default=None,
                    type=str,
                    help="Trained RoI model weights.")
    parser.add_argument("--output_dir",
                    default="../vimacsa",
                    type=str)
    
    # other parameters
    parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

    parser.add_argument("--get_cate",
                        action='store_true',
                        help="Whether to get image category.")
    
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=8.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    
    args = parser.parse_args()
    print("===================== RUN ROI CATEGORIES =====================")

    if args.no_cuda:
        device = 'cpu'
    else:
       device = 'cuda'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.get_cate:
        raise ValueError("At least one of `do_train` or `get_cate` must be True.")

    ASPECT = ['food', 'room', 'facilities', 'service', 'public_area'] # Our predefined aspect category

    if args.do_train:
        if args.roi_label_path == None:
           raise ValueError("Please provide annotated RoI file.")

        roi_df = pd.read_csv(f"{args.roi_label_path}")
        train_data, dev_test_data = train_test_split(roi_df,test_size=0.3,random_state=18)
        dev_data, test_data = train_test_split(dev_test_data,test_size=0.5,random_state=18)

        train_data = train_data.reset_index().drop('index',axis=1)
        dev_data = dev_data.reset_index().drop('index',axis=1)
        test_data = test_data.reset_index().drop('index',axis=1)

        train_set = RoiDataset(train_data,args.image_dir,ASPECT)
        dev_set = RoiDataset(dev_data,args.image_dir,ASPECT)
        test_set = RoiDataset(test_data,args.image_dir,ASPECT)

        train_loader = DataLoader(train_set,batch_size=args.train_batch_size,shuffle = True)
        dev_loader = DataLoader(dev_set,batch_size=args.eval_batch_size,shuffle = False)
        test_loader = DataLoader(test_set,batch_size=args.eval_batch_size,shuffle = False)

        num_train_steps = len(train_loader)*args.num_train_epochs

        model = MyRoIModel(len(ASPECT)) # No Location
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)
        max_accracy = 0.0

        logger.info("*************** Running training ***************")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: "+ str(train_idx) + " **********")
            logger.info("  Num examples = %d", train_data.shape[0])
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)

            model.train()
            with tqdm(train_loader, position=0, leave=True, desc="Iteration") as tepoch:
                for step, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {train_idx}")
                    input = batch['image']
                    label = batch['label']

                    input = input.to(device)
                    label = label.to(device)
                    logits = model(input)
                    
                    loss = criterion(logits,label)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    tepoch.set_postfix(loss=loss.item())

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", dev_data.shape[0])
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            idx2asp = {i:v for i,v in enumerate(ASPECT)}

            eval_epoch_loss = 0
            all_truth = []
            all_pred = []
            for step, batch in enumerate(tqdm(dev_loader, position=0, leave=True, desc="Evaluating")):
                input = batch['image']
                label = batch['label']

                input = input.to(device)
                label = label.to(device)
                
                with torch.no_grad():
                    logits = model(input)
                    loss = criterion(logits,label)
                    eval_epoch_loss += loss.item()
                    logits = logits.cpu().numpy()
                    pred = np.argmax(logits, axis = -1)

                    all_pred.extend(pred.tolist())
                    all_truth.extend(label.cpu().numpy().tolist())

            if step == 0:
                step = 1
            eval_epoch_loss /= step 

            logger.info("***** Precision, Recall, F1-score, Accuracy for each Aspect *****")
            all_precision, all_recall, all_f1 = macro_f1(all_truth, all_pred)
            matrix = confusion_matrix(all_truth, all_pred,labels = [i for i in range(len(ASPECT))])
            all_accuracy = matrix.diagonal()/matrix.sum(axis=1)
            all_accuracy = np.where(np.isnan(all_accuracy), 0, all_accuracy)


            for id_asp in range(len(ASPECT)):
                p_at_asp = all_precision[id_asp]
                r_at_asp = all_recall[id_asp]
                f1_at_asp = all_f1[id_asp]
                acc_at_asp = all_accuracy[id_asp]

                aspect_rs = {idx2asp[id_asp]:[p_at_asp,r_at_asp,f1_at_asp,acc_at_asp]}

                for key in sorted(aspect_rs.keys()):
                    logger.info("  %s = %s", key, str(aspect_rs[key]))

            all_precision = all_precision.mean()
            all_recall = all_recall.mean()
            all_f1 = all_f1.mean()
            all_accuracy = all_accuracy.mean()

            eval_accuracy = all_accuracy
            results = {'eval_loss': eval_epoch_loss,
                        'precision_score':all_precision,
                        'recall_score':all_recall,
                        'f_score': all_f1,
                        'accuracy': all_accuracy
                        }
            logger.info("***** Dev Eval results *****")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))

            if eval_accuracy >= max_accracy:
                # Save a trained model    
                save_model(f'{args.output_dir}/seed_{args.seed}_roi_model.pth',model,train_idx)
                max_accracy = eval_accuracy

        # TEST
        output_test_file = os.path.join(args.output_dir, "test_roi_results.txt")
        with open(output_test_file, "a") as writer:
            writer.write("***** Running evaluation on Test Set *****\n")
            writer.write(f"  Num examples = {test_data.shape[0]}\n" )
            writer.write(f"  Batch size = {args.eval_batch_size}\n")

            logger.info("***** Running evaluation on Test Set *****")
            logger.info("  Num examples = %d", test_data.shape[0])
            logger.info("  Batch size = %d", args.eval_batch_size)

        checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_roi_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_epoch_loss = 0
        test_all_truth = []
        test_all_pred = []

        for step, batch in enumerate(tqdm(test_loader, position=0, leave=True, desc="Evaluating")):
                input = batch['image']
                label = batch['label']

                input = input.to(device)
                label = label.to(device)
                
                with torch.no_grad():
                    logits = model(input)
                    loss = criterion(logits,label)
                    test_epoch_loss += loss.item()
                    logits = logits.cpu().numpy()
                    pred = np.argmax(logits, axis = -1)

                    test_all_pred.extend(pred.tolist())
                    test_all_truth.extend(label.cpu().numpy().tolist())

        if step == 0:
            step = 1
        test_epoch_loss /= step 

        with open(output_test_file, "a") as writer:
            logger.info("***** Precision, Recall, F1-score, Accuracy for each Aspect *****")
            writer.write("***** Precision, Recall, F1-score, Accuracy for each Aspect *****\n")

        test_all_precision, test_all_recall, test_all_f1 = macro_f1(test_all_truth, test_all_pred)
        matrix = confusion_matrix(test_all_truth, test_all_pred,labels = [i for i in range(len(ASPECT))])
        test_all_accuracy = matrix.diagonal()/matrix.sum(axis=1)
        test_all_accuracy = np.where(np.isnan(test_all_accuracy), 0, test_all_accuracy)

        for id_asp in range(len(ASPECT)):
            p_at_asp = test_all_precision[id_asp]
            r_at_asp = test_all_recall[id_asp]
            f1_at_asp = test_all_f1[id_asp]
            acc_at_asp = test_all_accuracy[id_asp]

            aspect_rs = {idx2asp[id_asp]:[p_at_asp,r_at_asp,f1_at_asp,acc_at_asp]}

            with open(output_test_file, "a") as writer:
                for key in sorted(aspect_rs.keys()):
                    logger.info("  %s = %s", key, str(aspect_rs[key]))
                    writer.write(f"{key} = {str(aspect_rs[key])}\n")

        test_all_precision = test_all_precision.mean()
        test_all_recall = test_all_recall.mean()
        test_all_f1 = test_all_f1.mean()
        test_all_accuracy = test_all_accuracy.mean()

        results = {'eval_loss': test_epoch_loss,
                    'precision_score':test_all_precision,
                    'recall_score':test_all_recall,
                    'f_score': test_all_f1,
                    'accuracy': test_all_accuracy
                    }
        
        with open(output_test_file, "a") as writer:
            logger.info("***** Test Eval results *****")
            writer.write("***** Test Eval results *****\n")

            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write(f"{key} = {str(results[key])}\n")

    if args.get_cate:
        print("===================== GET ROI CATEGORIES =====================")

        model = MyRoIModel(len(ASPECT)) # No Location
        model = model.to(device)

        if args.do_train:
            checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_roi_model.pth')
        else:
            try:
                checkpoint = load_model(args.weight_path)
            except:
                raise ValueError("Wrong image model weights!!!!")
        
        model.load_state_dict(checkpoint['model_state_dict'])

        ASPECT = np.asarray(ASPECT)

        image_label_dict = {}

        roi_df = pd.read_csv(f"{args.roi_label_path}")
        list_img_name = roi_df['file_name'].unique()
        
        for img_name in list_img_name:
            p = img_name + ".png"
            
            image = torchvision.io.read_image(os.path.join(args.image_dir,p),mode = torchvision.io.ImageReadMode.RGB)
            df_roi = roi_df[roi_df['file_name']==img_name][:6].reset_index()
            num_roi = df_roi.shape[0]

            image_aspect = []
            for i in range(num_roi):
                x1 = df_roi.loc[i,'x1']
                x2 = df_roi.loc[i,'x2']
                y1 = df_roi.loc[i,'y1']
                y2 = df_roi.loc[i,'y2']
                
                roi_img = image[:,x1:x2,y1:y2]
                roi_img = convert_img_to_tensor(roi_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(roi_img)
                    pred = np.argmax(pred.cpu().numpy(),axis=-1)
                    image_aspect.append(ASPECT[pred][0])
            image_aspect = list(set(image_aspect))
            print(f'At image {p}: {image_aspect}')
            image_label_dict[p] = image_aspect
            
        with open(f"{args.output_dir}/resnet152_roi_label.json", "x",encoding='utf-8') as f:
            json.dump(image_label_dict, f,indent = 2,ensure_ascii=False)

if __name__ == "__main__":
   main()