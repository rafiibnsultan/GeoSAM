import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
torch.cuda.is_available()

import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import h5py
import torch.nn.functional as F
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling import Sam
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast
import utils
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)

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

# Create a dataset
root_folder = '/home/rafi/tile2net/geoSAM_train/geoSAM_train/GT/Data/'      #define your dataset location here

dataset = CustomDataset(root_folder, transform=transform)

#the data loader
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#original parameters
sam_checkpoint = "/home/rafi/sam_vit_h_4b8939.pth"
# state_dicts = torch.load("/home/rafi/SAM/sam_decoder_multi_text.pth")

model_type = "vit_h"
model_save_path = '/home/rafi/SAM/'

#if you have cuda-based gpu
device = "cuda:0"

#model initialization
# torch.save(state_dicts['sam'], "/home/rafi/GeoSAM/sam_param.pth")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam = sam_model_registry[model_type](checkpoint="/home/rafi/GeoSAM/sam_param.pth").to(device=device)
sam.to(device=device)
sam.train()
predictor = SamPredictor(sam)

#implementation details


projection_layer = nn.Linear(512, 256).to(device)
optimizer = torch.optim.AdamW(list(sam.mask_decoder.parameters()) + list(projection_layer.parameters()), lr=1e-5, weight_decay=0.1)


seg_loss = monai.losses.DiceFocalLoss(to_onehot_y=True, softmax=True, squared_pred=True, reduction='mean')

num_epochs = 30
losses = []
best_loss = 1e10

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)        
classes = [0, 29, 75]       # 0=background, 29 = pedestrian, 75 = road, (it depends on the original color of the masks in the gts)
class_names = ["Background","Sidewalk and crosswalk", "Roads"]
if chatGPT==False:
    mod_cls_txt_encoding = torch.load("/home/rafi/GeoSAM/mod_cls_txt_encoding.pth").to(device)

input_size = (1024,1024)
original_size = (1024,1024)
mask_threshold = 0.0


# train
for epoch in range(num_epochs):
    epoch_loss = 0
    for idx, (images, masks, gts, image_names) in enumerate(tqdm(dataloader)):
        for i in range(images.shape[0]):
            image = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask = (masks[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
            gt = (gts[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
            
            
        
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
                        
                with torch.no_grad():
                    # gt_channel = gt[:, :, channel]
                    predictor.set_image(image)
                    image_embedding = predictor.get_image_embedding()
                
                
                left_clicks, right_clicks = utils.get_random_points(mask,classes[channel])
                all_points = np.concatenate((left_clicks, right_clicks), axis=0)
                all_points = np.array(all_points)
                point_labels = np.array([1]*left_clicks.shape[0] + [0]*right_clicks.shape[0], dtype=int)
                    
                if len(all_points) == 0:         #if the model can't generate any sparse prompts
                    transform = ResizeLongestSide(sam.image_encoder.img_size)

                    if embeddings!=None:
                        embeddings = embeddings.unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
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
                        
                    mask_predictions, _ = sam.mask_decoder(
                        image_embeddings=image_embedding.to(device), 
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings, 
                        dense_prompt_embeddings=dense_embeddings, 
                        multimask_output=False,
                    )
                    mask_predictions = F.interpolate(mask_predictions, image.shape[:2], mode="bilinear", align_corners=False)
                    
                    # mask_predictions = sam.postprocess_masks(mask_predictions, input_size, original_size)
                    # mask_predictions = mask_predictions > mask_threshold
                    
                    predictions.append(mask_predictions)
                else:
                 
                    transform = ResizeLongestSide(sam.image_encoder.img_size)
            
                    all_points = transform.apply_coords(all_points, (image.shape[0], image.shape[0])) 
                    all_points = torch.as_tensor(all_points, dtype=torch.float, device=device)
                    point_labels = torch.as_tensor(point_labels, dtype=torch.float, device=device)
                    all_points, point_labels = all_points[None, :, :], point_labels[None, :]
                    points = (all_points, point_labels)
                    
                    
                    if embeddings!=None:
                        embeddings = embeddings.unsqueeze(0).unsqueeze(0)
                        
                        
                            
                    
            
                    with torch.no_grad():
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
                    # predicted masks
                    mask_predictions, _ = sam.mask_decoder(
                        image_embeddings=image_embedding.to(device), 
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings, 
                        dense_prompt_embeddings=dense_embeddings, 
                        multimask_output=False,
                    )
                    mask_predictions = F.interpolate(mask_predictions, image.shape[:2], mode="bilinear", align_corners=False)
                    
                    # mask_predictions = sam.postprocess_masks(mask_predictions, input_size, original_size)
                    # mask_predictions = mask_predictions > mask_threshold
                    
                    
                    predictions.append(mask_predictions)
             
       
            predictions = torch.cat(predictions, dim=1)
            
        
        gts = utils.categorize(gts)       # Categorize the label to 0,1,2 for one hot encoding
        
        
        loss = seg_loss(predictions, gts.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    epoch_loss /= idx
    
    losses.append(epoch_loss)
    scheduler.step()
    print(f'EPOCH: {epoch+1}, Loss: {epoch_loss}')

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        if text == True:
            torch.save({
                'projection_layer': projection_layer.state_dict(),
                'sam': sam.state_dict(),
            }, join(model_save_path, 'sam_decoder_multi_text_gpt.pth'))
        else:
            torch.save({
                'projection_layer': projection_layer.state_dict(),
                'sam': sam.state_dict(),
            }, join(model_save_path, 'sam_decoder_multi_without_text.pth'))
        print("Saving weights, epoch: ", epoch+1)
