import os

import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from nets.SegResnet_enn import SegResNetENN
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ConcatItemsd,
    ToTensord,
    ScaleIntensityd,
    Activations,
    AsDiscrete,
)

transforms = Compose(
    [
        LoadImaged(keys=["pet", "ct", "mask"]),
        ScaleIntensityd(keys=["pet", "ct"]),
        AddChanneld(keys=["pet", "ct", "mask"]),
        ConcatItemsd(keys=["pet", "ct"], name="pet_ct", dim=0),
        ToTensord(keys=["pet_ct", "mask"]),
    ]
)
post_trans = Compose([Activations(softmax=False), AsDiscrete(threshold=0.5)])

model_params = {
    "in_channels": 2,
    "out_channels": 2,
    "spatial_dims": 3,
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SegResNetENN(model_params).to(device)


trained_model_path = r"./models/trained_model.pth"

pretrained_weights = torch.load(trained_model_path, map_location=device)
model.load_state_dict(pretrained_weights)

ct_file_path = r'./data/1_M0057CTres.nii'
pet_file_path = r'./data/1_M0057SUV.nii'
mask_file_path = r'./data/1_M0057SEG.nii'
single_sample = {
    "pet": pet_file_path,
    "ct": ct_file_path,
    "mask": mask_file_path,
}
transformed_sample = transforms(single_sample)


model.eval()
with torch.no_grad():
    input, label = transformed_sample["pet_ct"].unsqueeze(0), transformed_sample["mask"].unsqueeze(0)
    if torch.cuda.is_available():
        input, label = input.cuda(), label.cuda()
    output = model(input)
    output = output[:, :2, :, :, :] + 0.5 * output[:, 2, :, :, :].unsqueeze(1)
    output = post_trans(output)

output_tumor_channel = output[:, 1, :, :, :]
output_tumor_channel = output_tumor_channel.squeeze()
label = label.squeeze()

classification_tensor = torch.zeros_like(label, device=label.device)

classification_tensor[(output_tumor_channel == 1) & (label == 1)] = 1  # TP
classification_tensor[(output_tumor_channel == 1) & (label == 0)] = 2  # FP
classification_tensor[(output_tumor_channel == 0) & (label == 1)] = 3  # FN

image = sitk.ReadImage(ct_file_path)
image_array = sitk.GetArrayFromImage(image)
normalized_array = np.clip(image_array, -160, 240)
normalized_array = (normalized_array + 160) / 400
classification_array = classification_tensor.cpu().numpy()
rgb_array = np.zeros((classification_array.shape[0], classification_array.shape[1], classification_array.shape[2], 3))
gray_scale = np.clip(normalized_array, 0, 1)
gray_scale_transposed = np.transpose(gray_scale, (1, 2, 0))

tumor_mask = (label == 1).cpu().numpy()
highlighted_image = np.zeros((normalized_array.shape[1], normalized_array.shape[2], normalized_array.shape[0], 3))

for i in range(3):
    rgb_array[..., i] = gray_scale_transposed
    highlighted_image[..., i] = gray_scale_transposed

highlighted_image[tumor_mask, 0] = 1
highlighted_image[tumor_mask, 1] = 0
highlighted_image[tumor_mask, 2] = 0

colors = {
    1: [0, 0, 1],  # blue
    2: [0, 1, 0],  # green
    3: [1, 0, 0],  # red
}

for label, color in colors.items():
    for c in range(3):
        rgb_array[classification_array == label, c] = color[c]

slice_idx = 64
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("CT slice with Tumor Highlighted")
plt.imshow(highlighted_image[:, :, slice_idx, :])
plt.subplot(1, 2, 2)
plt.title("CT slice with ENN-SegResNet detected")
plt.imshow(rgb_array[:, :, slice_idx, :])
plt.show()
