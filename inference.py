## road = 75, sidewalk/crosswalk = 29, background = 0
import torch
#print(torch.cuda.is_available())

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
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

import utils

class CustomDataset(Dataset):
    def __init__(self, root_dir, embeddings_dir, transform=None):
        self.embeddings_file = embeddings_dir
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')     #these are the pseudo-labels, generated from the secondary CNN image encoder (for us it's Tile2Net)
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
        
        
        with h5py.File(self.embeddings_file, 'r') as hdf_file:
            dataset = hdf_file['data']
            retrieved_row = dataset[idx, :]
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        gt = Image.open(gt_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            gt = self.transform(gt)
        
        return image, mask, gt, image_name, retrieved_row
    

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create a dataset
root_folder = '/home/rafi/tile2net/geoSAM/geoSAM/GT/Data/'      #define your dataset location here
embeddings_file = root_folder+"image_embeddings.h5"             #this is the embeddings of each of the files to be tested and saved into a h5 file
dataset = CustomDataset(root_folder, embeddings_file, transform=transform)

#the data loader
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

sam_checkpoint = "/home/rafi/Dropbox/SAM/sam_decoder_multi.pth" #this is the fine-tuned SAM decoder you get after you done the training
model_type = "vit_h"        

#if you have cuda-based gpu
device = "cuda:1"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)

#defining you model
predictor = SamPredictor(sam)

#defining one-hot encoding
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(threshold=128)

#your evaluation metric, you can change it to anything else
mIOU_metric_batch = MeanIoU(include_background=False, reduction="mean_batch", get_not_nans=True)

#inference loop
classes = [0, 29, 75] # 0=background, 29 = pedestrian, 75 = road, (it depends on the original color of the masks in the gts)
for idx, (images, masks, gts, image_names, embeddings_batch) in enumerate(tqdm(dataloader)):
    for i in range(images.shape[0]):
        image = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = (masks[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
        gt = (gts[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        if np.all(gt == 0):         #if the gt doesn't have any of the class we need we just ignore it
            continue
        
        embeddings = embeddings_batch[i].cpu().numpy()
        image_name = image_names[i]

        
        predictions = []

        for channel in range(len(classes)):
            predictor.set_image(image)
            # print(image_names[i])
            
            left_clicks, right_clicks = utils.get_random_points(mask,classes[channel])
            #sparse prompts
            all_points = np.concatenate((left_clicks, right_clicks), axis=0)
            all_points = np.array(all_points)
            point_labels = np.array([1]*left_clicks.shape[0] + [0]*right_clicks.shape[0], dtype=int)
            
            
            if len(all_points)==0:      #if the model can't generate any sparse prompts
                temp_mask, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                mask_input=None,
                embeddings = embeddings,
                multimask_output=True,
                )
                
                input = logits[np.argmax(scores), :, :]     #dense prompts
                output_mask, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    mask_input=input[None, :, :],
                    embeddings = None,
                    multimask_output=False,
                )
                
                #post-processing
                output_mask_processed = utils.post_process_segmentation_map(output_mask)
                predictions.append(torch.from_numpy(output_mask_processed.squeeze()))
                
            else:
                temp_mask, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                mask_input=None,
                embeddings = embeddings,
                multimask_output=True,
                )

                input = logits[np.argmax(scores), :, :]     #dense prompts

                output_mask, scores, logits = predictor.predict(
                    point_coords=all_points,
                    point_labels=point_labels,
                    mask_input=input[None, :, :],
                    embeddings = None,
                    multimask_output=False,
                )
    
                output_mask_processed = utils.post_process_segmentation_map(output_mask)

                
                predictions.append(torch.from_numpy(output_mask_processed.squeeze()))
        
        predictions = torch.stack(predictions).unsqueeze(0)
    
        #for calculating mIoU
        predictions = [post_pred(i) for i in decollate_batch(predictions)]
        gts = [post_label(i) for i in decollate_batch(utils.categorize(gts))]
        
        mIOU_metric_batch(y_pred=predictions, y=gts)
        
metric_batch = mIOU_metric_batch.aggregate()
print(f"IoU of Sidewalk/Crosswalk: {metric_batch[0][0].item()}, IoU of road: {metric_batch[0][1].item()}")

                