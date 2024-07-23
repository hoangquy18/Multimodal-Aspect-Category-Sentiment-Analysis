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
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from torchvision.models import resnet152,ResNet152_Weights,resnet50, ResNet50_Weights
import random
import logging

logging.basicConfig(filename='image_categories.log',format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self, data,root_dir):
        self.image_label = data
        self.root_dir = root_dir
    def __len__(self):
        return self.image_label.shape[0]
    def __getitem__(self,index):
        image_name = self.image_label.loc[index, "file_name"]
        
        image = torchvision.io.read_image(os.path.join(self.root_dir,image_name),mode = torchvision.io.ImageReadMode.RGB)

        label = torch.from_numpy(self.image_label.iloc[index,2:].values.astype(int)) # no location

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
            "label": label
        }
    
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
      = precision_recall_fscore_support(y_true, y_pred, average='macro',zero_division=0.0,labels = [0,1])
    return p_macro, r_macro, f_macro

def convert_img_to_tensor(root_dir, img_path):
    image = torchvision.io.read_image(os.path.join(root_dir,img_path),mode = torchvision.io.ImageReadMode.RGB)

    transforms = v2.Compose([
                        v2.Resize((224,224),antialias=True),  # args.crop_size, by default it is set to be 224
                        # v2.RandomHorizontalFlip(),
                        v2.ConvertImageDtype(torch.float32),
                        v2.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
                                    ])

    image = transforms(image)

    return image

def predict_wrapper(model, list_aspect, root_dir, img_path,device):
    img = convert_img_to_tensor(root_dir, img_path)
    pred = model(img.unsqueeze(0).to(device))
    pred = torch.sigmoid(pred).squeeze(0)
    pred = pred > 0.45 # in our case, threshold equal 0.45
    pred = pred.cpu().numpy().astype(int)
    pred = np.where(pred==1)[0]
    
    return list(list_aspect[pred])
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",
                        default='../image',
                        type=str,
                        required=True,
                        help="The image data dir. Should contain the list of images files for the MACSA task.")
    parser.add_argument("--image_label_path",
                    default=None,
                    type=str,
                    help="The labeled image. Should contain the list of annotated images files for the MACSA task.")

    parser.add_argument("--weight_path",
                    default=None,
                    type=str,
                    help="Trained image model weights.")
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
    print("===================== RUN IMAGE CATEGORIES =====================")

    if args.no_cuda:
        device = 'cpu'
    else:
       device = 'cuda'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.get_cate:
        raise ValueError("At least one of `do_train` or `get_cate` must be True.")

    ASPECT = ['Food', 'Room', 'Facilities', 'Service', 'Public_area'] # Our predefined aspect category

    if args.do_train:
        if args.image_label_path == None:
           raise ValueError("Please provide annotated image file.")

        image_label = pd.read_excel(args.image_label_path)
        image_label = image_label.fillna(0)
        image_label = image_label.loc[~(image_label.iloc[:,1:]==0).all(axis=1)]
        image_label = image_label.reset_index().drop("index",axis=1)

        train_data, dev_test_data = train_test_split(image_label,test_size=0.3,random_state=18)
        dev_data, test_data = train_test_split(dev_test_data,test_size=0.5,random_state=18)

        train_data = train_data.reset_index().drop('index',axis=1)
        dev_data = dev_data.reset_index().drop('index',axis=1)
        test_data = test_data.reset_index().drop('index',axis=1)

        train_set = ImageDataset(train_data,args.image_dir)
        dev_set = ImageDataset(dev_data,args.image_dir)
        test_set = ImageDataset(test_data,args.image_dir)

        train_loader = DataLoader(train_set,batch_size=args.train_batch_size,shuffle = True)
        dev_loader = DataLoader(dev_set,batch_size=args.eval_batch_size,shuffle = False)
        test_loader = DataLoader(test_set,batch_size=args.eval_batch_size,shuffle = False)

        num_train_steps = len(train_loader)*args.num_train_epochs

        model = MyImgModel(len(ASPECT)) # No Location
        model = model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
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
                    
                    loss = criterion(logits,label.float())

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    tepoch.set_postfix(loss=loss.item())

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", dev_data.shape[0])
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            idx2asp = {i:v for i,v in enumerate(ASPECT)}
            true_label_list = {asp:[] for asp in ASPECT}
            pred_label_list = {asp:[] for asp in ASPECT}

            eval_epoch_loss = 0
            for step, batch in enumerate(tqdm(dev_loader, position=0, leave=True, desc="Evaluating")):
                input = batch['image']
                label = batch['label']

                input = input.to(device)
                label = label.to(device)
                
                with torch.no_grad():
                    logits = model(input)
                    loss = criterion(logits,label.float())
                    eval_epoch_loss += loss.item()
                    logits = torch.sigmoid(logits)

                    logits = logits.cpu().numpy()

                    for id_asp in range(len(ASPECT)):
                        asp_label = label[:,id_asp].cpu().numpy()
                        
                        pred = np.asarray(logits[:,id_asp] > 0.7).astype(int) # multi-label threshold 

                        true_label_list[idx2asp[id_asp]].append(asp_label)
                        pred_label_list[idx2asp[id_asp]].append(pred)

            if step == 0:
                step = 1
            eval_epoch_loss /= step 

            logger.info("***** Precision, Recall, F1-score, Accuracy for each Aspect *****")
            all_precision, all_recall, all_f1 = 0, 0, 0
            all_accuracy = 0
            for id_asp in range(len(ASPECT)):
                tr = np.concatenate(true_label_list[idx2asp[id_asp]])
                pr = np.concatenate(pred_label_list[idx2asp[id_asp]])

                precision, recall, f1_score = macro_f1(tr,pr)
                accuracy = accuracy_score(tr,pr)
                aspect_rs = {idx2asp[id_asp]:[precision,recall,f1_score,accuracy]}

                for key in sorted(aspect_rs.keys()):
                    logger.info("  %s = %s", key, str(aspect_rs[key]))

                all_precision += precision
                all_recall += recall
                all_f1 += f1_score
                all_accuracy += accuracy

            all_precision /= len(ASPECT)
            all_recall /= len(ASPECT)
            all_f1 /= len(ASPECT)
            all_accuracy /= len(ASPECT)

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
                save_model(f'{args.output_dir}/seed_{args.seed}_image_model.pth',model,train_idx)
                max_accracy = eval_accuracy

        output_test_file = os.path.join(args.output_dir, "test_image_results.txt")
        with open(output_test_file, "a") as writer:
            writer.write("***** Running evaluation on Test Set *****\n")
            writer.write(f"  Num examples = {test_data.shape[0]}\n" )
            writer.write(f"  Batch size = {args.eval_batch_size}\n")

            logger.info("***** Running evaluation on Test Set *****")
            logger.info("  Num examples = %d", test_data.shape[0])
            logger.info("  Batch size = %d", args.eval_batch_size)

        checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_image_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_epoch_loss = 0

        for step, batch in enumerate(tqdm(test_loader, position=0, leave=True, desc="Evaluating")):
            input = batch['image']
            label = batch['label']

            input = input.to(device)
            label = label.to(device)
            
            with torch.no_grad():
                logits = model(input)
                loss = criterion(logits,label.float())
                test_epoch_loss += loss.item()
                logits = torch.sigmoid(logits)

                logits = logits.cpu().numpy()

                for id_asp in range(len(ASPECT)):
                    asp_label = label[:,id_asp].cpu().numpy()
                    
                    pred = np.asarray(logits[:,id_asp] > 0.7).astype(int) # multi-label threshold 

                    true_label_list[idx2asp[id_asp]].append(asp_label)
                    pred_label_list[idx2asp[id_asp]].append(pred)

        if step == 0:
            step = 1
        test_epoch_loss /= step 

        with open(output_test_file, "a") as writer:
            logger.info("***** Precision, Recall, F1-score, Accuracy for each Aspect *****")
            writer.write("***** Precision, Recall, F1-score, Accuracy for each Aspect *****\n")

        test_all_precision, test_all_recall, test_all_f1 = 0, 0, 0
        test_all_accuracy = 0
        for id_asp in range(len(ASPECT)):
            tr = np.concatenate(true_label_list[idx2asp[id_asp]])
            pr = np.concatenate(pred_label_list[idx2asp[id_asp]])

            precision, recall, f1_score = macro_f1(tr,pr)
            accuracy = accuracy_score(tr,pr)
            aspect_rs = {idx2asp[id_asp]:[precision,recall,f1_score,accuracy]}

            with open(output_test_file, "a") as writer:
                for key in sorted(aspect_rs.keys()):
                    logger.info("  %s = %s", key, str(aspect_rs[key]))
                    writer.write(f"{key} = {str(aspect_rs[key])}\n")

            test_all_precision += precision
            test_all_recall += recall
            test_all_f1 += f1_score
            test_all_accuracy += accuracy

        test_all_precision /= len(ASPECT)
        test_all_recall /= len(ASPECT)
        test_all_f1 /= len(ASPECT)
        test_all_accuracy /= len(ASPECT)

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
        print("===================== GET IMAGE CATEGORIES =====================")

        model = MyImgModel(len(ASPECT)) # No Location
        model = model.to(device)

        if args.do_train:
            checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_image_model.pth')
        else:
            try:
                checkpoint = load_model(args.weight_path)
            except:
                raise ValueError("Wrong image model weights!!!!")
        
        model.load_state_dict(checkpoint['model_state_dict'])

        ASPECT = np.asarray(ASPECT)

        image_path_list = os.listdir(args.image_dir)
        img_label_dict = {}

        for img_path in image_path_list:
            lb = predict_wrapper(model, ASPECT,args.image_dir, img_path,device)
            print(f"At image {img_path}: {lb}")
            img_label_dict[img_path] = lb

        # Save result
        with open(f"{args.output_dir}/resnet152_image_label.json", "x",encoding='utf-8') as f:
            json.dump(img_label_dict, f,indent = 2,ensure_ascii=False)

if __name__ == "__main__":
   main()