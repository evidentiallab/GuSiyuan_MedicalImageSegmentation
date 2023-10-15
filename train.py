import os.path

# from model import *
import torch
import monai
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ConcatItemsd,
    ToTensord,
    ScaleIntensityd,
)
from monai.data import Dataset, list_data_collate
from glob import glob

data_dir = r"C:\Users\lifel\Projects\Evidential-neural-network-for-lymphoma-segmentation\Evidential_segmentation\LYMPHOMA\Data"
pet_dir = os.path.join(data_dir, 'SUV')
ct_dir = os.path.join(data_dir, 'CTres')
mask_dir = os.path.join(data_dir, 'SEG')

pet_files = sorted(glob(os.path.join(pet_dir, '*SUV.nii')))
ct_files = sorted(glob(os.path.join(ct_dir, '*CTres.nii')))
mask_files = sorted(glob(os.path.join(mask_dir, '*SEG.nii')))

data_dicts = [
    {"pet": pet_file, "ct": ct_file, "mask": mask_file}
    for pet_file, ct_file, mask_file in zip(pet_files, ct_files, mask_files)
]

transforms = Compose(
    [
        LoadImaged(keys=["pet", "ct", "mask"]),
        ScaleIntensityd(keys=["pet", "ct"]),
        AddChanneld(keys=["pet", "ct", "mask"]),
        ConcatItemsd(keys=["pet", "ct"], name="pet_ct", dim=0),
        ToTensord(keys=["pet_ct", "mask"]),
    ]
)

dataset = Dataset(data=data_dicts, transform=transforms)

dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=list_data_collate)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        kernel_size=5,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,).to(device)

# params = model.parameters()
# params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
loss_function = monai.losses.DiceLoss(include_background=False, softmax=True, squared_pred=True, to_onehot_y=True)

epoch_loss_values = list()

for epoch in range(100):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{100}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in dataloader:
        step += 1
        inputs, labels = batch_data["pet_ct"].to(device), batch_data["mask"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(dataset) // dataloader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    scheduler.step(epoch_loss)
