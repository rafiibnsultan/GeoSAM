import numpy as np
import cv2
import torch
import h5py

import clip
import torch

import os
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = 'your-API-key'

def get_random_points(mask, mask_value, foreground_points=2000, background_points=1000):
    ## road = 75, sidewalk/crosswalk = 29, background = 0
    # Find the indices (coordinates) of the foreground pixels
    
    if (mask_value == 29):
        foreground_indices = np.argwhere((mask == 29) | (mask == 76))
        background_indices = np.argwhere((mask != 29) & (mask!=76))
    else:
        foreground_indices = np.argwhere(mask == mask_value)
        
        background_indices = np.argwhere(mask != mask_value)
    
    # Shuffle the indices to get random points
    np.random.shuffle(foreground_indices)
    np.random.shuffle(background_indices)
    
    
    # Swap the indices before saving
    foreground_indices = foreground_indices[:, [1, 0]]
    background_indices = background_indices[:, [1, 0]]
    
    # left clicks
    try:
        left_clicks = foreground_indices[np.random.choice(foreground_indices.shape[0], foreground_points, replace=False)]
        
    except:
        left_clicks = np.empty(0)
        
    try: 
        right_clicks = background_indices[np.random.choice(background_indices.shape[0], background_points, replace=False)]
    except:
        right_clicks = np.empty(0)
    # left_clicks = np.array(foreground_indices[:foreground_points])
    # right_clicks = np.array(background_indices[:background_points])

    # left_clicks = [(point[1], point[0]) for point in left_clicks]
    if len(left_clicks) == 0 or len(right_clicks) == 0:
        return (np.empty(0), np.empty(0))
    
    return (left_clicks, right_clicks)


def post_process_segmentation_map(segmentation_map):
    segmentation_map = segmentation_map.squeeze()
    segmentation_map = segmentation_map.astype(np.uint8) * 255
    # Define the structuring element (you can experiment with size and shape)
    
    # Perform dilation
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    processed_mask = cv2.dilate(segmentation_map, structuring_element, iterations=10)
    
    # Perform erosion
    # structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    processed_mask = cv2.erode(processed_mask, structuring_element, iterations=10)  
    
    return np.expand_dims(processed_mask, axis=0)

def categorize(gts):
    # Create a mapping from unique values to integers
    value_to_int = {v.item(): i for i, v in enumerate(torch.unique(gts))}

    # Apply the mapping
    for original_value, mapped_value in value_to_int.items():
        gts[gts == original_value] = mapped_value

    # print(gts)
    return gts

def check_embeddings_file(file_path):
    # Load the HDF5 file
    with h5py.File(file_path, 'r') as hdf_file:
        if 'data' in hdf_file:
            dataset = hdf_file['data']
            num_rows = dataset.shape[0]
            print("Number of rows:", num_rows)
        else:
            print("Dataset 'data' not found in the HDF5 file.")
            
    row_number = 0  # Change this to the desired row number (0-indexed)
    with h5py.File(file_path, 'r') as hdf_file:
        dataset = hdf_file['data']
        retrieved_row = dataset[row_number, :]

        print("Retrieved Row:", retrieved_row)
        
    return(np.unique(retrieved_row))

def chatGPT_description(class_name, device):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative assistant, skilled in providing detailed visual descriptions of objects as seen in aerial imagery."},
            {"role": "user", "content": f"Print out a visual description (don't mention their names) that can be seen from aerial images of {class_name} (in one line, 4 to 5 words, not more, not less)."}
        ]
    )
    
    
    sentence = [f'{class_name}: {completion.choices[0].message.content}']
    # print(sentence)
    model, _ = clip.load('ViT-B/32', device)
    
    text_inputs = clip.tokenize(sentence).to(device)
    text_features = model.encode_text(text_inputs) 
    return text_features