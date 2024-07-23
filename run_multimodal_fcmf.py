import torch
from text_preprocess import *
from vimacsa_dataset import *
from fcmf_framework.fcmf_modeling import *
from sklearn.metrics import precision_recall_fscore_support
import argparse
import logging
import random
from tqdm.auto import tqdm,trange
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
import pandas as pd
from underthesea import word_tokenize,text_normalize
from fcmf_framework.resnet_utils import *
from torchvision.models import resnet152,ResNet152_Weights,resnet50, ResNet50_Weights
from fcmf_framework.optimization import BertAdam
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import json
from  torch.cuda.amp import autocast

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, _ \
      = precision_recall_fscore_support(y_true, y_pred, average='macro',zero_division=0.0)

    return p_macro, r_macro, f_macro

def save_model(path, model, epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    },path)

def load_model(path):
    check_point = torch.load(path)
    return check_point

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default='../vimacsa',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the MACSA task.")
    parser.add_argument("--list_aspect", 
                        "--names-list", 
                        default=['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area'],
                        nargs='+', help = "List of predefined Aspect.")
    parser.add_argument("--num_polarity",
                        default=4,
                        type=int,
                        help="Number of sentiment polarity.")
    parser.add_argument("--num_imgs",
                    default=7,
                    type=int,
                    help="Number of images.")
    parser.add_argument("--num_rois",
                    default=4,
                    type=int,
                    help="Number of RoIs.")
    parser.add_argument('--image_dir', default='../vimacsa/image', help='path to images')

    parser.add_argument("--pretrained_model", default=None, type=str, required=True,
                    help="Pre-trained model selected in the list: vinai/phobert-base, "
                    "vinai/phobert-base-v2, xlm-roberta-base, xlm-roberta-large, uitnlp/visobert")
    
    parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
    
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=170,
                        type=int,
                        help="The maximum total input sequence length after tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
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
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--num_train_epochs",
                        default=8.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--ddp",
                        action='store_true',
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--add_layer', action='store_true', help='whether to add another encoder layer')

    args = parser.parse_args()



    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    elif args.ddp:
        assert torch.cuda.is_available()
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0
    elif not args.ddp:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Running on device:{ddp_local_rank}")
    if master_process:
        print("===================== RUN Fine-grained Cross-modal Fusion =====================")
        logging.basicConfig(filename=f'training_fcmf.log',format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO)

        logger = logging.getLogger(__name__)

        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, ddp_world_size, bool(args.ddp ), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    if args.train_batch_size == 0:
        raise ValueError("train_batch_size must greater than gradient_accumulation_steps")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if ddp_world_size > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.distributed.init_process_group(backend='nccl')

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    except:
        raise ValueError("Wrong pretrained model.")
    

    normalize_class = TextNormalize()
    ASPECT = args.list_aspect
    # print(ASPECT)
    try:
        roi_df = pd.read_csv(f"{args.data_dir}/roi_data.csv")
        roi_df['file_name'] = roi_df['file_name'] + '.png'
    except:
        raise ValueError("Can't find roi_data.csv")
    
    try:
        with open(f'{args.data_dir}/resnet152_image_label.json') as imf:
            dict_image_aspect = json.load(imf)

        with open(f'{args.data_dir}/resnet152_roi_label.json') as rf:
            dict_roi_aspect = json.load(rf)
    except:
        raise ValueError("Get image/roi aspect category first. Please run run_image_categories.py or run_roi_categories.py")
    
    if args.do_train:
        train_data = pd.read_json(f'{args.data_dir}/train.json')
        dev_data = pd.read_json(f'{args.data_dir}/dev.json')
        train_data['comment'] = train_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
        dev_data['comment'] = dev_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))

        num_splitted_train = train_data.shape[0] // ddp_world_size
        train_data = train_data.iloc[num_splitted_train*ddp_local_rank: num_splitted_train*ddp_local_rank + num_splitted_train,:]

        # num_splitted_dev = dev_data.shape[0] // ddp_world_size
        # dev_data = dev_data.iloc[num_splitted_dev*ddp_local_rank: num_splitted_dev*ddp_local_rank + num_splitted_dev,:]

        num_train_steps = int(
            train_data.shape[0] / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        train_dataset = MACSADataset(train_data,tokenizer,f"{args.image_dir}",roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)
        dev_dataset = MACSADataset(dev_data,tokenizer,f"{args.image_dir}",roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)

    model = FCMF(pretrained_path = args.pretrained_model,
                 num_labels = args.num_polarity,
                 num_imgs = args.num_imgs,
                 num_roi = args.num_rois)
    
    img_res_model = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res_model = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2).to(device)

    resnet_img = myResNetImg(resnet = img_res_model, if_fine_tune = args.fine_tune_cnn, device = device)
    resnet_roi = myResNetRoI(resnet = roi_res_model, if_fine_tune = args.fine_tune_cnn, device = device)

    model.to(device)
    resnet_img.to(device)
    resnet_roi.to(device)

    if args.ddp:
        model = torch.nn.DataParallel(model, device_ids = [ddp_local_rank])
        resnet_img = torch.nn.DataParallel(resnet_img, device_ids = [ddp_local_rank])
        resnet_roi = torch.nn.DataParallel(resnet_roi, device_ids = [ddp_local_rank])

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    criterion = torch.nn.CrossEntropyLoss()

    if args.fp16:
        mode = 'FP16'
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = True if mode == 'TF32' else False
        scaler = torch.cuda.amp.GradScaler(enabled=True if mode == 'FP16' else False)

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                        lr=args.learning_rate)
    
    global_step = 0
    if args.do_train:
        if not args.ddp:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        dev_sampler = SequentialSampler(dev_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.eval_batch_size)

        num_train_steps = len(train_dataloader)*args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_train_steps*args.warmup_proportion, num_training_steps=num_train_steps)

        max_f1 = 0.0
        if master_process:
            logger.info("*************** Running training ***************")
        for train_idx in range(int(args.num_train_epochs)):
            # print(f"At epoch {train_idx}")
            if master_process:
                logger.info("********** Epoch: "+ str(train_idx) + " **********")
                logger.info("  Num examples = %d", train_data.shape[0])
                logger.info("  Batch size = %d", args.train_batch_size)
                logger.info("  Num steps = %d", num_train_steps)
            
            model.train()
            resnet_img.train()
            resnet_roi.train()
            resnet_img.zero_grad()
            resnet_roi.zero_grad()

            with tqdm(train_dataloader, position = 0, leave=True) as tepoch:
                for step, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {train_idx}")

                    t_img_features, roi_img_features, roi_coors, \
                    all_input_ids, all_token_types_ids, all_attn_mask, \
                    all_added_input_mask, all_label_id  = batch

                    roi_img_features = roi_img_features.float()
                    
                    roi_coors = roi_coors.to(device)
                    t_img_features = t_img_features.to(device)
                    roi_img_features = roi_img_features.to(device)

                    all_input_ids = all_input_ids.to(device)
                    all_token_types_ids = all_token_types_ids.to(device)
                    all_attn_mask = all_attn_mask.to(device)

                    all_added_input_mask = all_added_input_mask.to(device)
                    all_label_id = all_label_id.to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16 if args.fp16 else torch.float32, enabled=True if args.fp16 else False):
                        with torch.no_grad():
                            encoded_img = []
                            encoded_roi = []

                            for img_idx in range(args.num_imgs):
                                img_features = resnet_img(t_img_features[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1) # batch_size, 49, 2048
                                encoded_img.append(img_features)

                                roi_f = []
                                for roi_idx in range(args.num_rois):
                                    roi_features = resnet_roi(roi_img_features[:,img_idx,roi_idx,:]).squeeze(1) # batch_size, 1, 2048
                                    roi_f.append(roi_features)

                                roi_f = torch.stack(roi_f,dim=1)
                                encoded_roi.append(roi_f)

                            encoded_img = torch.stack(encoded_img,dim=1) # batch_size, num_img, 49, 2048   
                            encoded_roi = torch.stack(encoded_roi,dim=1) # batch_size, num_img, num_roi, 49,2048

                        all_asp_loss = 0
                        for id_asp in range(len(ASPECT)):
                            logits = model(
                                        input_ids = all_input_ids[:,id_asp,:], token_type_ids = all_token_types_ids[:,id_asp,:], attention_mask = all_attn_mask[:,id_asp,:], 
                                        added_attention_mask = all_added_input_mask[:,id_asp,:],
                                        visual_embeds_att = encoded_img,
                                        roi_embeds_att = encoded_roi,
                                        roi_coors = roi_coors
                                        )
                            loss = criterion(logits,all_label_id[:,id_asp])
                            if ddp_world_size > 1:
                                loss = loss.mean()
                            all_asp_loss += loss

                        lll = all_asp_loss.mean().item()

                        if args.gradient_accumulation_steps > 1:
                            all_asp_loss = all_asp_loss / args.gradient_accumulation_steps

                        if args.ddp:
                            model.require_backward_grad_sync = (step == args.gradient_accumulation_steps - 1)
                            dist.all_reduce(all_asp_loss.detach(), op = dist.ReduceOp.AVG)

                        if args.fp16:
                            scaler.scale(all_asp_loss).backward()
                        else:
                            all_asp_loss.backward()
                        
                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                            if args.fp16:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            
                            torch.cuda.synchronize() # wait other gpus finished
                            scheduler.step()
                            optimizer.zero_grad()
                            global_step += 1

                        tepoch.set_postfix(loss=lll)

            if master_process:
                logger.info("***** Running evaluation on Dev Set*****")
                logger.info("  Num examples = %d", dev_data.shape[0])
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                resnet_img.eval()
                resnet_roi.eval()

                eval_loss, eval_f1 = 0, 0

                idx2asp = {i:v for i,v in enumerate(ASPECT)}
                epoch_loss = 0

                true_label_list = {asp:[] for asp in ASPECT}
                pred_label_list = {asp:[] for asp in ASPECT}

                for step, batch in enumerate(tqdm(dev_dataloader, position=0, leave=True, desc="Evaluating")):

                # for i, data in tqdm(enumerate(dataloader)):
                    t_img_features, roi_img_features, roi_coors, \
                    all_input_ids, all_token_types_ids, all_attn_mask, \
                    all_added_input_mask, all_label_id  = batch
                    
                    roi_img_features = roi_img_features.float()

                    roi_coors = roi_coors.to(device)
                    t_img_features = t_img_features.to(device)
                    roi_img_features = roi_img_features.to(device)

                    all_input_ids = all_input_ids.to(device)
                    all_token_types_ids = all_token_types_ids.to(device)
                    all_attn_mask = all_attn_mask.to(device)

                    all_added_input_mask = all_added_input_mask.to(device)
                    all_label_id = all_label_id.to(device)
                    
                    with torch.no_grad():
                        encoded_img = []
                        encoded_roi = []

                        for img_idx in range(args.num_imgs):
                            img_features = resnet_img(t_img_features[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1) # batch_size, 49, 2048
                            encoded_img.append(img_features)

                            roi_f = []
                            for roi_idx in range(args.num_rois):
                                roi_features = resnet_roi(roi_img_features[:,img_idx,roi_idx,:]).squeeze(1) # batch_size, 1, 2048
                                roi_f.append(roi_features)

                            roi_f = torch.stack(roi_f,dim=1)
                            encoded_roi.append(roi_f)

                        encoded_img = torch.stack(encoded_img,dim=1) # batch_size, num_img, 49, 2048   
                        encoded_roi = torch.stack(encoded_roi,dim=1) # batch_size, num_img, num_roi, 49,2048

                        all_asp_loss = 0
                        for id_asp in range(len(ASPECT)):
                            logits = model(
                                    input_ids = all_input_ids[:,id_asp,:], token_type_ids = all_token_types_ids[:,id_asp,:], attention_mask = all_attn_mask[:,id_asp,:], 
                                    added_attention_mask = all_added_input_mask[:,id_asp,:],
                                    visual_embeds_att = encoded_img,
                                    roi_embeds_att = encoded_roi,
                                    roi_coors = roi_coors
                            )

                            eval_loss = criterion(logits,all_label_id[:,id_asp])

                            all_asp_loss += eval_loss
                            asp_label = all_label_id[:,id_asp].to('cpu').numpy()
                            logits = logits.detach().cpu().numpy()
                            pred = np.argmax(logits,axis = -1)

                            true_label_list[idx2asp[id_asp]].append(asp_label)
                            pred_label_list[idx2asp[id_asp]].append(pred)

                        epoch_loss += all_asp_loss.mean().item()
                
                if step == 0:
                    step = 1
                epoch_loss /= step 

                all_precision, all_recall, all_f1 = 0, 0, 0
                logger.info("***** Precision, Recall, F1-score for each Aspect *****")

                for id_asp in range(len(ASPECT)):
                    tr = np.concatenate(true_label_list[idx2asp[id_asp]])
                    pr = np.concatenate(pred_label_list[idx2asp[id_asp]])

                    precision, recall, f1_score = macro_f1(tr,pr)

                    aspect_rs = {idx2asp[id_asp]:[precision,recall,f1_score]}
                    for key in sorted(aspect_rs.keys()):
                        logger.info("  %s = %s", key, str(aspect_rs[key]))

                    all_precision += precision
                    all_recall += recall
                    all_f1 += f1_score

                all_precision /= len(ASPECT)
                all_recall /= len(ASPECT)
                all_f1 /= len(ASPECT)

                eval_f1 = all_f1

                results = {'eval_loss': epoch_loss,
                            'precision_score':all_precision,
                            'recall_score':all_recall,
                            'f_score': all_f1,
                            }
                # print(f"Dev: Precision = {all_precision}, Recall = {all_recall}, F1-score = {all_f1}")
                logger.info("***** Dev results *****")
                for key in sorted(results.keys()):
                    logger.info("  %s = %s", key, str(results[key]))

                if eval_f1 >= max_f1:
                    # Save a trained model    
                    if args.do_train:
                        save_model(f'{args.output_dir}/seed_{args.seed}_fcmf_model.pth',model,train_idx)
                        save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model.pth',resnet_img,train_idx)
                        save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model.pth',resnet_roi,train_idx)

                    max_f1 = eval_f1


    if args.do_eval and (not args.ddp or torch.distributed.get_rank() == 0):
        # Load a trained model that highest f1-score
        model_checkpoint = load_model(f"{args.output_dir}/seed_{args.seed}_fcmf_model.pth")
        resimg_checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_resimg_model.pth')
        resroi_checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_resroi_model.pth')

        img_res_model = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2).to(device)
        roi_res_model = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2).to(device)
        resnet_img = myResNetImg(resnet = img_res_model, if_fine_tune = args.fine_tune_cnn, device = device)
        resnet_roi = myResNetRoI(resnet = roi_res_model, if_fine_tune = args.fine_tune_cnn, device = device)

        model = FCMF(pretrained_path = args.pretrained_model,
                    num_labels = args.num_polarity,
                    num_imgs = args.num_imgs,
                    num_roi = args.num_rois)

        model.load_state_dict(model_checkpoint['model_state_dict'])
        resnet_img.load_state_dict(resimg_checkpoint['model_state_dict'])
        resnet_roi.load_state_dict(resroi_checkpoint['model_state_dict'])

        model = model.to(device)
        resnet_img = resnet_img.to(device)
        resnet_roi = resnet_roi.to(device)

        # LOAD TEST EVAL
        test_data = pd.read_json(f'{args.data_dir}/test.json')
        test_data['comment'] = test_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))

        test_dataset = MACSADataset(test_data,tokenizer,f"{args.image_dir}",roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

        output_eval_file = os.path.join(args.output_dir, "test_results.txt")

        with open(output_eval_file, "a") as writer:
            logger.info("***** Running evaluation on Test Set*****")
            logger.info("  Num examples = %d", test_data.shape[0])
            logger.info("  Batch size = %d", args.eval_batch_size)

            writer.write("***** Running evaluation on Test Set *****\n")
            writer.write(f"  Num examples = {test_data.shape[0]}\n" )
            writer.write(f"  Batch size = {args.eval_batch_size}\n")

        model.eval()
        resnet_img.eval()
        resnet_roi.eval()

        eval_loss, eval_f1 = 0, 0

        idx2asp = {i:v for i,v in enumerate(ASPECT)}
        epoch_loss = 0

        true_label_list = {asp:[] for asp in ASPECT}
        pred_label_list = {asp:[] for asp in ASPECT}

        for step, batch in enumerate(tqdm(test_dataloader, position=0, leave=True, desc="Evaluating")):

            t_img_features, roi_img_features, roi_coors, \
            all_input_ids, all_token_types_ids, all_attn_mask, \
            all_added_input_mask, all_label_id  = batch
            
            roi_img_features = roi_img_features.float()

            roi_coors = roi_coors.to(device)
            t_img_features = t_img_features.to(device)
            roi_img_features = roi_img_features.to(device)

            all_input_ids = all_input_ids.to(device)
            all_token_types_ids = all_token_types_ids.to(device)
            all_attn_mask = all_attn_mask.to(device)

            all_added_input_mask = all_added_input_mask.to(device)
            all_label_id = all_label_id.to(device)
            
            with torch.no_grad():
                encoded_img = []
                encoded_roi = []

                for img_idx in range(args.num_imgs):
                    img_features = resnet_img(t_img_features[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1) # batch_size, 49, 2048
                    encoded_img.append(img_features)

                    roi_f = []
                    for roi_idx in range(args.num_rois):
                        roi_features = resnet_roi(roi_img_features[:,img_idx,roi_idx,:]).squeeze(1) # batch_size, 1, 2048
                        roi_f.append(roi_features)

                    roi_f = torch.stack(roi_f,dim=1)
                    encoded_roi.append(roi_f)

                encoded_img = torch.stack(encoded_img,dim=1) # batch_size, num_img, 49, 2048   
                encoded_roi = torch.stack(encoded_roi,dim=1) # batch_size, num_img, num_roi, 49,2048

                all_asp_loss = 0
                for id_asp in range(len(ASPECT)):
                    logits = model(
                            input_ids = all_input_ids[:,id_asp,:], token_type_ids = all_token_types_ids[:,id_asp,:], attention_mask = all_attn_mask[:,id_asp,:], 
                            added_attention_mask = all_added_input_mask[:,id_asp,:],
                            visual_embeds_att = encoded_img,
                            roi_embeds_att = encoded_roi,
                            roi_coors = roi_coors
                    )

                    eval_loss = criterion(logits,all_label_id[:,id_asp])

                    all_asp_loss += eval_loss
                    asp_label = all_label_id[:,id_asp].to('cpu').numpy()
                    logits = logits.detach().cpu().numpy()
                    pred = np.argmax(logits,axis = -1)

                    true_label_list[idx2asp[id_asp]].append(asp_label)
                    pred_label_list[idx2asp[id_asp]].append(pred)

                epoch_loss += all_asp_loss.mean().item()
        
        if step == 0:
            step = 1
        epoch_loss /= step 

        all_precision, all_recall, all_f1 = 0, 0, 0

        with open(output_eval_file, "a") as writer:
            logger.info("***** Precision, Recall, F1-score for each Aspect *****")
            writer.write("***** Precision, Recall, F1-score for each Aspect *****\n")

        for id_asp in range(len(ASPECT)):
            tr = np.concatenate(true_label_list[idx2asp[id_asp]])
            pr = np.concatenate(pred_label_list[idx2asp[id_asp]])

            precision, recall, f1_score = macro_f1(tr,pr)

            aspect_rs = {idx2asp[id_asp]:[precision,recall,f1_score]}

            with open(output_eval_file, "a") as writer:
                for key in sorted(aspect_rs.keys()):
                    logger.info("  %s = %s", key, str(aspect_rs[key]))
                    writer.write(f"{key} = {str(aspect_rs[key])}\n")

            all_precision += precision
            all_recall += recall
            all_f1 += f1_score

        all_precision /= len(ASPECT)
        all_recall /= len(ASPECT)
        all_f1 /= len(ASPECT)

        eval_f1 = all_f1

        results = {'eval_loss': epoch_loss,
                    'precision_score':all_precision,
                    'recall_score':all_recall,
                    'f_score': all_f1,
                    }
        # print(f"Test: Precision = {all_precision}, Recall = {all_recall}, F1-score = {all_f1}")

        with open(output_eval_file, "a") as writer:
            logger.info("***** Test results *****")
            writer.write("***** Test results *****\n")

            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write(f"{key} = {str(results[key])}\n")

if __name__ =='__main__':
    main()




