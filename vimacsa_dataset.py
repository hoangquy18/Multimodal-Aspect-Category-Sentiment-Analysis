from torchvision.io import read_image, ImageReadMode
import torch
from torchvision import transforms
from transformers import AutoTokenizer
import os 
import numpy as np

# by default: num_img = num_roi = 7
# return:
#   [CLS] aspect category [SEP] text [SEP] list image aspect (split by , ) [SEP] list roi aspect (split by ,) [SEP]
#   image: (num_img, 3, 224, 224)
#   roi: (num_img, num_roi, 3, 224, 224) 
class MACSADataset(torch.utils.data.Dataset):
    def __init__(self, data,tokenizer, img_folder, roi_df, dict_image_aspect,dict_roi_aspect, num_img, num_roi):
        self.data = data
        self.ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']

        self.pola_to_num = {
            "None": 0,
            "Negative": 1,
            "Neutral": 2,
            "Positive": 3
        }

        self.transform = transforms.Compose([
                            transforms.Resize((224,224),antialias=True),  # args.crop_size, by default it is set to be 224
                            # transforms.RandomHorizontalFlip(),
                            transforms.ConvertImageDtype(torch.float32),
                            transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))
                                        ])
        self.roi_df = roi_df
        self.img_folder = img_folder
        self.dict_image_aspect = dict_image_aspect
        self.dict_roi_aspect = dict_roi_aspect
        self.num_img = num_img
        self.num_roi = num_roi
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):

        idx_data = self.data.iloc[idx,:].values

        text = idx_data[0]
        
        # img = os.path.join(self.img_folder,idx_data[2])
        list_image_aspect = []
        list_roi_aspect = []
        for img_name in idx_data[1]:
            try:
                list_image_aspect.extend(self.dict_image_aspect[img_name])
            except:
                pass
            try:
                list_roi_aspect.extend(self.dict_roi_aspect[img_name])
            except:
                pass
        list_image_aspect = list(set(list_image_aspect))
        list_roi_aspect = list(set(list_roi_aspect))

        if len(list_image_aspect) == 0:
            list_image_aspect = ['empty']
        if len(list_roi_aspect) == 0:
            list_roi_aspect = ['empty']
        # list_image_aspect = list(map(lambda x: x.lower(),list_image_aspect))
        # list_roi_aspect = list(map(lambda x: x.lower(),list_roi_aspect))

        text_img_label = idx_data[3]
        list_aspect, list_polar = [],[]
        for asp_pol in text_img_label:
            asp,pol = asp_pol.split("#")
            if "_" in asp:
                asp = "Public area"
            list_aspect.append(asp)
            list_polar.append(pol)

        for asp in self.ASPECT:
            if "_" in asp:
                asp = "Public area"
            if asp not in list_aspect:
                list_aspect.append(asp)
                list_polar.append('None')
        
        all_input_ids = []
        all_token_types_ids = []
        all_attn_mask = []
        all_label_id = []
        all_added_input_mask = []

        for ix in range(len(self.ASPECT)):
            asp = self.ASPECT[ix]
            if "_" in asp:
                asp = "Public area"

            idx_asp_in_list_asp = list_aspect.index(asp)

            joined_aspect = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}"
            joined_aspect = joined_aspect.lower().replace('_',' ')

            combine_text = f"{asp} </s></s> {text}"
            combine_text = combine_text.lower().replace('_',' ')
            tokens = self.tokenizer(combine_text, joined_aspect, max_length=170,truncation='only_first',padding='max_length', return_token_type_ids=True)

            input_ids = torch.tensor(tokens['input_ids'])
            token_type_ids = torch.tensor(tokens['token_type_ids'])
            attention_mask = torch.tensor(tokens['attention_mask'])
            added_input_mask =torch.tensor( [1] * (170+49))

            label_id = list_polar[idx_asp_in_list_asp]

            all_input_ids.append(input_ids)
            all_token_types_ids.append(token_type_ids)
            all_attn_mask.append(attention_mask)
            all_added_input_mask.append(added_input_mask)
            all_label_id.append(self.pola_to_num[label_id])

        all_input_ids = torch.stack(all_input_ids,dim=0)
        all_token_types_ids = torch.stack(all_token_types_ids)
        all_attn_mask = torch.stack(all_attn_mask)
        all_added_input_mask = torch.stack(all_added_input_mask)

        all_label_id = torch.tensor(all_label_id)

        list_img_path = idx_data[1]
        # os_list_img_path = [os.path.join(self.img_folder,path) for path in list_img_path]
        
        list_img_features = []
        global_roi_features = [] # num_img, num_roi, 3, 224, 224
        global_roi_coor = []
        for img_path in list_img_path:
          image_os_path = os.path.join(self.img_folder, img_path)
          one_image = read_image(image_os_path, mode = ImageReadMode.RGB)
          img_transform = self.transform(one_image).unsqueeze(0) # 1, 3, 224, 224
          list_img_features.append(img_transform)

          ##### ROI
          list_roi_img = [] # num_roi, 3, 224, 224
          list_roi_coor = [] # num_roi, 4
          roi_in_img_df = self.roi_df[self.roi_df['file_name'] == img_path][:self.num_roi]
        #   print(roi_in_img_df)
          if roi_in_img_df.shape[0] == 0:
              list_roi_img = np.zeros((self.num_roi,3,224,224))
              
            #   print(len(list_roi_img))
              global_roi_coor.append(np.zeros((self.num_roi,4)))
              global_roi_features.append(list_roi_img)
              continue
          
          for i_roi in range(roi_in_img_df.shape[0]):
            x1, x2, y1, y2 = roi_in_img_df.iloc[i_roi,1:4+1].values            

            roi_in_image = one_image[:,x1:x2,y1:y2]
            roi_transform = self.transform(roi_in_image).numpy() # 3, 224, 224

            x1, x2, y1, y2 = x1/512, x2/512, y1/512, y2/512
            cv = lambda x: np.clip([x],0.0,1.0)[0]
            x1 = cv(x1)
            x2 = cv(x2)
            y1 = cv(y1)
            y2 = cv(y2) 

            list_roi_coor.append([x1,x2,y1,y2])
            list_roi_img.append(roi_transform)

        #   print("For loop first:", len(list_roi_img))
          if i_roi < self.num_roi:
            for k in range(self.num_roi - i_roi-1):
                list_roi_img.append(np.zeros((3,224,224)))
                list_roi_coor.append(np.zeros((4,)))

          global_roi_features.append(list_roi_img)
          global_roi_coor.append(list_roi_coor)

        ### FOR IMAGE
        t_img_features = torch.zeros((self.num_img, 3, 224, 224))
        num_imgs = len(list_img_features)

        if num_imgs >= self.num_img:
            for i in range(self.num_img):
                t_img_features[i,:] = list_img_features[i]
        else:
            for i in range(self.num_img):
                if i < num_imgs:
                    # img_features[(self.max_img_len-num_imgs)+i,:] = imgs[i]
                    t_img_features[i, :] = list_img_features[i]
                else:
                    break
        
        ### FOR ROI
        global_roi_features = np.asarray(global_roi_features)
        global_roi_coor = np.asarray(global_roi_coor)

        roi_img_features = np.zeros((self.num_img, self.num_roi, 3, 224, 224))
        roi_coors = np.zeros((self.num_img,self.num_roi,4))

        num_img_roi = len(global_roi_features)

        if num_img_roi >= self.num_img:
            for i in range(self.num_img):
                roi_img_features[i,:] = global_roi_features[i]
                roi_coors[i,:] = global_roi_coor[i]
        else:
            for i in range(self.num_img):
                if i < num_imgs:
                    # img_features[(self.max_img_len-num_imgs)+i,:] = imgs[i]
                    roi_img_features[i, :] = global_roi_features[i]
                    roi_coors[i,:] = global_roi_coor[i,:]
                else:
                    break

        roi_img_features = torch.tensor(roi_img_features)
        roi_coors = torch.tensor(roi_coors)

        return t_img_features, roi_img_features, roi_coors, all_input_ids, all_token_types_ids, all_attn_mask, all_added_input_mask, all_label_id
