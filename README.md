# ViMACSA: Vietnamese Multimodal Aspect-Category Sentiment Analysis
![](images/overview_fcmf.png)

# Dataset
Our ViMACSA dataset comprises 4,876 documents and 14,000 images. Each document is accompanied by up to 7 images. This dataset is constructed with the goal of recognizing both explicit aspects and implicit aspects in the document.

<p align="center">
  <img src="images/ex_data.png" />
</p>


To understand more about the dataset, please read this paper: [New Benchmark Dataset and Fine-Grained Cross-Modal Fusion Framework for Vietnamese Multimodal Aspect-Category Sentiment Analysis
](https://arxiv.org/abs/2405.00543) 

Our dataset is used only for research purposes. Download our ViMACSA dataset directly at: https://drive.google.com/file/d/1OjWwzdbhvhYc864Tpt6Xw9anBLfgNwmt/view?usp=sharing

## Dataset statistics
![The overview statistics of ViMACSA dataset](images/dataset_stat.png)
*Table 1. The overview statistics of ViMACSA dataset.*

# Running The Code
## Install Requirements
```
pip install -r requirements.txt
```
## Get Image/RoI Aspect Category
### Image
```
python image_processing/run_image_categories.py 
      --image_dir path_to_image_folder
      --image_label_path path_to_image_label #all_image_label.xlsx 
      --output_dir test_image 
      --do_train 
      --get_cate #whether to get image category
```
### RoI
```
python image_processing/run_roi_categories.py 
      --image_dir path_to_image_folder 
      --roi_label_path path_to_roi_label #test_roi_data.csv 
      --output_dir test_image 
      --do_train 
      --get_cate #whether to get RoI category
```

## Training FCMF Framework
```
!torchrun --standalone --nproc_per_node=n_gpu run_multimodal_fcmf.py
        --data_dir data_folder_dir
        --list_aspect Location Food Room Facilities Service Public_area 
        --num_polarity 4 --num_imgs 1 --num_rois 4
        --image_dir path_to_image_folder
        --pretrained_model vinai/phobert-base 
        --output_dir model_output 
        --train_batch_size 8  --eval_batch_size 8 
        --num_train_epochs 8
        --learning_rate 3e-5 
        --warmup_proportion 0.1 --gradient_accumulation_steps 2 
        --do_train
        --fp16 
        --ddp 
```
# Compare with different baseline models.
![Experiment results on the ViMACSA dataset](images/exper.png)                   
*Table 2. Experiment results on the ViMACSA dataset.*

# Citation
Please cite the following paper if you use this dataset:
```bibtex
@article{nguyen2024new,
  title={New Benchmark Dataset and Fine-Grained Cross-Modal Fusion Framework for Vietnamese Multimodal Aspect-Category Sentiment Analysis},
  author={Nguyen, Quy Hoang and Nguyen, Minh-Van Truong and Van Nguyen, Kiet},
  journal={arXiv preprint arXiv:2405.00543},
  year={2024}
}
```

# Contact
If you have any questions, please feel free to contact nhq188@gmail.com.
