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

import utils
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)

class CustomDataset(Dataset):
    def __init__(self, root_dir, embeddings_dir, transform=None):
        self.embeddings_file = embeddings_dir
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
root_folder = '/home/rafi/tile2net/geoSAM_train/geoSAM_train/GT/Data/'      #define your dataset location here
embeddings_file = embeddings_file = root_folder+"image_embeddings.h5"       #this is the embeddings of each of the files to be tested and saved into a h5 file
dataset = CustomDataset(root_folder, embeddings_file, transform=transform)

#the data loader
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#original parameters
sam_checkpoint = "/home/rafi/sam_vit_h_4b8939.pth"
model_type = "vit_h"
model_save_path = '/home/rafi/Dropbox/SAM/'

#if you have cuda-based gpu
device = "cuda:0"

#model initialization
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.train()
predictor = SamPredictor(sam)

#implementation details
optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=1e-5, weight_decay=0.1)
seg_loss = monai.losses.DiceFocalLoss(to_onehot_y=True, softmax=True, squared_pred=True, reduction='mean')

num_epochs = 100
losses = []
best_loss = 1e10

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)        
classes = [0, 29, 75]       # 0=background, 29 = pedestrian, 75 = road, (it depends on the original color of the masks in the gts)

# train
for epoch in range(num_epochs):
    epoch_loss = 0
    for idx, (images, masks, gts, image_names, embeddings_batch) in enumerate(tqdm(dataloader)):
        for i in range(images.shape[0]):
            image = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask = (masks[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
            gt = (gts[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
            embeddings = embeddings_batch[i].cpu().numpy()
            image_name = image_names[i]
           
            predictions = []
            for channel in range(len(classes)):
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
                    embeddings = embeddings.reshape((1, 256, 64, 64))   #fixed size as SAM accepts
                    embeddings_torch = torch.as_tensor(embeddings, dtype=torch.float, device=device)
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    dense_embeddings = embeddings_torch
                
                    mask_predictions, _ = sam.mask_decoder(
                        image_embeddings=image_embedding.to(device), 
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings, 
                        dense_prompt_embeddings=dense_embeddings, 
                        multimask_output=False,
                    )
                    mask_predictions = F.interpolate(mask_predictions, image.shape[:2], mode="bilinear", align_corners=False)
                    predictions.append(mask_predictions)
                else:
                 
                    transform = ResizeLongestSide(sam.image_encoder.img_size)
                    embeddings = embeddings.reshape((1, 256, 64, 64))
                    embeddings_torch = torch.as_tensor(embeddings, dtype=torch.float, device=device)
                    with torch.no_grad():
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                        )
                    dense_embeddings = embeddings_torch
            
                    mask_predictions, _ = sam.mask_decoder(
                        image_embeddings=image_embedding.to(device), 
                        image_pe=sam.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=sparse_embeddings, 
                        dense_prompt_embeddings=dense_embeddings, 
                        multimask_output=False,
                    )
            
                    all_points = transform.apply_coords(all_points, (image.shape[0], image.shape[0])) 
                    all_points = torch.as_tensor(all_points, dtype=torch.float, device=device)
                    point_labels = torch.as_tensor(point_labels, dtype=torch.float, device=device)
                    all_points, point_labels = all_points[None, :, :], point_labels[None, :]
                        
                    points = (all_points, point_labels)
            
                    with torch.no_grad():
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points = points,
                            boxes=None,
                            masks=mask_predictions,
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
        torch.save(sam.state_dict(), join(model_save_path, 'sam_decoder_multi.pth'))
        print("Saving weights, epoch: ", epoch+1)
