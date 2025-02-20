import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.lane_detector import LaneDetector
from models.enet import ENet
import os
import json
import torch
"""
testQ8.py is just a copy of test_lane_detection.py that 
"""

# Define dataset and checkpoint paths
DATASET_PATH = "/opt/data/TUSimple/test_set"
CHECKPOINT_PATH = "checkpoints/enet_checkpoint_epoch_49.pth"  # Path to the trained model checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the ENet model
def load_enet_model(checkpoint_path, device="cuda"):
    enet_model = ENet(binary_seg=2, embedding_dim=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    enet_model.load_state_dict(checkpoint['model_state_dict'])
    enet_model.eval()
    return enet_model

def perspective_transform(image):
    """
    Transform an image into a bird's eye view.
        1. Calculate the image height and width.
        2. Define source points on the original image and corresponding destination points.
        3. Compute the perspective transform matrix using cv2.getPerspectiveTransform.
        4. Warp the original image using cv2.warpPerspective to get the transformed output.
    """
    
    ####################### TODO: Your code starts Here #######################
    height, width = image.shape[:2]
    
    # top left, bottom left, bottom right, top right
    dst_pts = np.float32([
        [0, 0], 
        [0, height], 
        [width, height],      
        [width, 0]         
    ])
    # width, height
    ## 67,716,506,350,788,355,1202,716
    src_pts = np.float32([
        [(506./1280) * width, (350./720) * height],  
        [(67./1280) * width, (716./720) * height], 
        [(1202./1280) * width, (716./720) * height],  
        [(788./1280) * width, (355./720) * height]  
    ])
    
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    transformed_image = cv2.warpPerspective(image, M, (width, height))
    ####################### TODO: Your code ends Here #######################
    
    return transformed_image


# Function to visualize lane predictions for multiple images in a single row
def visualize_lanes_row(images, instances_maps, alpha=0.7):
    """
    Visualize lane predictions for multiple images in a single row
    For each image:
        1. Resize it to 512 x 256 for consistent visualization.
        2. Apply perspective transform to both the original image and its instance map.
        3. Overlay the instance map to a plot with the corresponding original image using a specified alpha value.
    """
    
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    ####################### TODO: Your code starts Here #######################
        
    if num_images == 1:
        axes = [axes]  

    for i in range(num_images):
        image = cv2.resize(images[i], (512, 256))
        # print(f'original image: {cv2.imshow("original image", image)}')
        instance_map = cv2.resize(instances_maps[i], (512, 256))
        transformed_image = perspective_transform(image)
        transformed_map = perspective_transform(instance_map)

        # normalize
        # print(f'straight up max of image: {np.max(np.squeeze(transformed_map))}')
        transformed_map /= np.max(np.squeeze(transformed_map))
        # print(f'transformed map values after division: {np.unique(transformed_map), transformed_map.shape}')
        transformed_map *= 255.0
        # print(f'instance map: {cv2.imshow("instance map", instance_map)}')
        print(f'transformed image: {cv2.imshow("transformed image", transformed_image)}')
        print(f'transformed map: {cv2.imshow("transformed map", transformed_map), transformed_map.shape}')
        # print(f'transformed map values: {np.unique(transformed_map), transformed_map.shape}')
        transformed_image = transformed_image.astype(np.uint8)
        transformed_map = transformed_map.astype(np.uint8)

        """
        transformed_map works but is grayscale. convert to color
        """
        if len(transformed_map.shape) == 2:
           transformed_map = cv2.cvtColor(transformed_map, cv2.COLOR_GRAY2BGR)
        
        colored_map = cv2.applyColorMap(transformed_map, cv2.COLORMAP_JET)
        # jet is full rainbow
        
        # print(f'colored map: {cv2.imshow("colored map", colored_map), colored_map.shape, colored_map}')
        
        overlayed = cv2.addWeighted(transformed_image, 1 - alpha, colored_map, alpha, 1)
        # print(f'overlayed: {cv2.imshow("overlayed", overlayed), overlayed.shape}')
        blended = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
        print(f'blended: {cv2.imshow("blended", blended), blended.shape}')

        axes[i].imshow(blended)
        axes[i].axis("off")

    ####################### TODO: Your code ends Here #######################

    plt.tight_layout()
    plt.show()

def main():
    # Initialize device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enet_model = load_enet_model(CHECKPOINT_PATH, device)
    lane_predictor = LaneDetector(enet_model, device=device)

    # # List of test image paths
    # sub_paths = [
    #     "clips/0530/1492626047222176976_0/20.jpg",
    #     "clips/0530/1492626286076989589_0/20.jpg",
    #     "clips/0531/1492626674406553912/20.jpg",
    #     "clips/0601/1494452381594376146/20.jpg",
    #     "clips/0601/1494452431571697487/20.jpg"
    # ]

    TRAIN_DATASET_PATH = "/opt/data/TUSimple/train_set/"

    # ground truth
    json_path = os.path.join(TRAIN_DATASET_PATH, "label_data_0531.json")
    data = []
    image_lookup = {}
    with open(json_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                temp = json.loads(line)
                data.append(temp)  # Parse each line as a dictionar
                image_lookup[temp['raw_file']] = temp
    
    # first_dict = data[0]
    # print(image_lookup.keys())

    # print("raw file: ", first_dict['raw_file'])

    # print(f"0531 json data first object: {len(first_dict['lanes'])}")
    # for i in range(4):
    #     print(first_dict['lanes'][i])

    #test_image_paths = [os.path.join(DATASET_PATH, sub_path) for sub_path in sub_paths]
    # image, predicted segmentation
    train_image_subpath = 'clips/0531/1492634660192614813/20.jpg' # arbitrarily chosen
    # if train_image_subpath in image_lookup:
    #     print(train_image_subpath, "in image_lookup")
    #     print(image_lookup[train_image_subpath])
    # predicted is what we alr get
    # idk lane embedding lol
    train_image_path = os.path.join(TRAIN_DATASET_PATH, train_image_subpath)

    print(f'train image path: {train_image_path}')
    # Load and process image

    image = cv2.imread(train_image_path)
    if image is None:
        print(f"Error: Unable to load image at {train_image_path}")
        return 0
    
    # get ground truth
    lane_img = image.copy()
    label = image_lookup[train_image_subpath] # lanes, h_samples, raw_file
    mask = np.zeros((720, 1280), dtype=np.uint8)  # Original resolution
    lanes = label['lanes']
    h_samples = label['h_samples']
    for lane in lanes:
        for x, y in zip(lane, h_samples):
            if x != -2:  # Skip invalid points
                cv2.circle(lane_img, (int(x), int(y)), radius=5, color=1, thickness=-1)
    print(f'lane  mask?: {cv2.imshow("lane mask", lane_img), lane_img.shape}')

    # lane embeddings?
    # images = [image]
    # images = images.to(DEVICE) # device???
    # binary_logits, instance_embeddings = enet_model(image)
    # print("embeddings?: ", instance_embeddings)

    # original image
    print(f'original image b4 input: {cv2.imshow("og", image), image.shape}')
    instances_map = lane_predictor(image)
    print(cv2.imshow("instances map before visualize", instances_map))
    visualize_lanes_row([image], [instances_map])

if __name__ == "__main__":
    main()
