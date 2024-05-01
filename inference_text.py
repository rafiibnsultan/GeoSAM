## road = 75, sidewalk/crosswalk = 29, background = 0
import torch
#print(torch.cuda.is_available())
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import average_precision_score
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import MeanIoU
from tqdm import tqdm
from torch.cuda.amp import autocast

import utils

text = True
chatGPT = True

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.gt_dir = os.path.join(root_dir, 'gt_multi')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))
        self.gt_filenames = sorted(os.listdir(self.gt_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]
        gt_name = self.gt_filenames[idx]
        
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        gt_path = os.path.join(self.gt_dir, gt_name)
        
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        gt = Image.open(gt_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            gt = self.transform(gt)
        
        return image, mask, gt, image_name
    

transform = transforms.Compose([
    transforms.ToTensor(),
])
#if you have cuda-based gpu
device = "cuda:0"

# Create a dataset
root_folder = '/home/rafi/tile2net/geoSAM/geoSAM/GT/Data/'      #define your dataset location here
# root_folder = '/home/rafi/tile2net/geoSAM_eva1024/geoSAM_eva1024/GT/Data/'


dataset = CustomDataset(root_folder, transform=transform)

#the data loader
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# loading saved weights
if text==True:
    if chatGPT == True:
        state_dicts = torch.load("/home/rafi/SAM/sam_decoder_multi_text_gpt.pth") #this is the fine-tuned SAM decoder you get after you done the training
    else:
        state_dicts = torch.load("/home/rafi/SAM/sam_decoder_multi_text.pth") #this is the fine-tuned SAM decoder you get after you done the training
else:
    state_dicts = torch.load("/home/rafi/SAM/sam_decoder_multi_without_text.pth") #this is the fine-tuned SAM decoder you get after you done the training
model_type = "vit_h"        
torch.save(state_dicts['sam'], "/home/rafi/GeoSAM/sam_param.pth")
sam = sam_model_registry[model_type](checkpoint="/home/rafi/GeoSAM/sam_param.pth").to(device=device)

projection_layer = nn.Linear(512, 256).to(device)
projection_layer.load_state_dict(state_dicts['projection_layer'])

#defining you model
predictor = SamPredictor(sam)

#defining one-hot encoding
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(threshold=128)

#your evaluation metric, you can change it to anything else
mIOU_metric_batch = MeanIoU(include_background=False, reduction="mean_batch", get_not_nans=True)

#inference loop
classes = [0, 29, 75] # 0=background, 29 = pedestrian, 75 = road, (it depends on the original color of the masks in the gts)
class_names = ["Background","Sidewalk and crosswalk", "Roads"]

if chatGPT==False:
    mod_cls_txt_encoding = torch.load("/home/rafi/GeoSAM/mod_cls_txt_encoding.pth").to(device)

input_size = (1024,1024)
original_size = (1024,1024)
mask_threshold = 0.0

mAP=[]
for idx, (images, masks, gts, image_names) in enumerate(tqdm(dataloader)):
    for i in range(images.shape[0]):
        image = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = (masks[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
        gt = (gts[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        if np.all(gt == 0):         #if the gt doesn't have any of the class we need we just ignore it
            continue
        
        image_name = image_names[i]

        
        predictions = []

        for channel in range(len(classes)):
            if channel == 0:
                    embeddings = None
            else:
                if chatGPT == True:
                    embeddings = utils.chatGPT_description(class_names[channel],device).to(device)
                    embeddings = embeddings.squeeze(0)
                    # print(embeddings.shape)                                 #torch.Size([512])  
                else:
                    embeddings = mod_cls_txt_encoding[0][channel-1]       #torch.Size([512])    
                with autocast():
                    embeddings = projection_layer(embeddings.half())    #torch.Size([256])
            predictor.set_image(image)
            image_embedding = predictor.get_image_embedding()
            # print(image_names[i])
            
            left_clicks, right_clicks = utils.get_random_points(mask,classes[channel])
            #sparse prompts
            all_points = np.concatenate((left_clicks, right_clicks), axis=0)
            all_points = np.array(all_points)
            point_labels = np.array([1]*left_clicks.shape[0] + [0]*right_clicks.shape[0], dtype=int)
            
            
            if len(all_points)==0:      #if the model can't generate any sparse prompts
                transform = ResizeLongestSide(sam.image_encoder.img_size)
                
                if embeddings!=None:
                    embeddings = embeddings.unsqueeze(0).unsqueeze(0)
                    
                if text==True:
                            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                                points=None,
                                boxes=embeddings,
                                masks=None,
                            )
                else:
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
            
            
                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=image_embedding.to(device), 
                    image_pe=sam.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=sparse_embeddings, 
                    dense_prompt_embeddings=dense_embeddings, 
                    multimask_output=False,
                )
  
             
                
                output_mask = sam.postprocess_masks(low_res_masks, input_size, original_size)
                output_mask = output_mask > mask_threshold
                output_mask = output_mask[0].detach().cpu().numpy()
                
                #post-processing
                output_mask_processed = utils.post_process_segmentation_map(output_mask)
                predictions.append(torch.from_numpy(output_mask_processed.squeeze()))
                
            else:
                transform = ResizeLongestSide(sam.image_encoder.img_size)
                
                if embeddings!=None:
                    embeddings = embeddings.unsqueeze(0).unsqueeze(0)
                
                all_points = transform.apply_coords(all_points, (image.shape[0], image.shape[0])) 
                all_points = torch.as_tensor(all_points, dtype=torch.float, device=device)
                point_labels = torch.as_tensor(point_labels, dtype=torch.float, device=device)
                all_points, point_labels = all_points[None, :, :], point_labels[None, :]
                points = (all_points, point_labels)  
                    
                if text==True:
                            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                                points=points,
                                boxes=embeddings,
                                masks=None,
                            )
                else:
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=points,
                        boxes=None,
                        masks=None,
                    )
                
            
                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=image_embedding.to(device), 
                    image_pe=sam.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=sparse_embeddings, 
                    dense_prompt_embeddings=dense_embeddings, 
                    multimask_output=False,
                )
                
                
                
                output_mask = sam.postprocess_masks(low_res_masks, input_size, original_size)
                output_mask = output_mask > mask_threshold
                
                output_mask = output_mask[0].detach().cpu().numpy()
                
                #post-processing
                output_mask_processed = utils.post_process_segmentation_map(output_mask)
                predictions.append(torch.from_numpy(output_mask_processed.squeeze()))
        
        predictions = torch.stack(predictions).unsqueeze(0)
    
        #for calculating mIoU
        predictions = [post_pred(i) for i in decollate_batch(predictions)]
        gts = [post_label(i) for i in decollate_batch(utils.categorize(gts))]
        
        mIOU_metric_batch(y_pred=predictions, y=gts)
        
        
        predictions = torch.stack(predictions).cpu().detach().numpy()
        gts = torch.stack(gts).cpu().detach().numpy()
        
        # #mAP
        # # Step 1: Extract relevant channels (excluding the background)
        # predictions = predictions[:, 1:, :, :].reshape(2, -1) # Shape: (1, 2, 1024, 1024)
        # gts = gts[:, 1:, :, :].reshape(2, -1)                # Shape: (1, 2, 1024, 1024)


        # # Step 3 & 4: Compute average precision score for each class
        # ap_scores = [average_precision_score(gts[i], predictions[i]) for i in range(2)]

        # mAP.append(ap_scores)
metric_batch = mIOU_metric_batch.aggregate()
print(f"IoU of Sidewalk/Crosswalk: {metric_batch[0][0].item()}, IoU of road: {metric_batch[0][1].item()}")

# mAP = np.array(mAP)
# print(f"mAP of Sidewalk/Crosswalk: {mAP[:, 0].mean()}, mAP of road: {mAP[:, 1].mean()}")

                