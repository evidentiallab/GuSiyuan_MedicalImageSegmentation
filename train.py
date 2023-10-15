import os.path
from glob import glob

import torch
import monai
from monai.data import Dataset, list_data_collate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ConcatItemsd,
    ToTensord,
    ScaleIntensityd,
)
from monai.networks.nets import UNet

import argparse
import logging
import datetime

# Configure logging with timestamp-based filename
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"training_{formatted_time}.log"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_filename)])
logger = logging.getLogger()


parser = argparse.ArgumentParser(description="Training script for medical image segmentation")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the data")
parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save trained models")
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

data_dir = args.data_dir
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
train_data_dicts, val_data_dicts = train_test_split(data_dicts, test_size=0.20, random_state=42)

transforms = Compose(
    [
        LoadImaged(keys=["pet", "ct", "mask"]),
        ScaleIntensityd(keys=["pet", "ct"]),
        AddChanneld(keys=["pet", "ct", "mask"]),
        ConcatItemsd(keys=["pet", "ct"], name="pet_ct", dim=0),
        ToTensord(keys=["pet_ct", "mask"]),
    ]
)

train_dataset = Dataset(data=train_data_dicts, transform=transforms)
val_dataset = Dataset(data=val_data_dicts, transform=transforms)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, collate_fn=list_data_collate)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, collate_fn=list_data_collate)

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
optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
loss_function = monai.losses.DiceLoss(include_background=False, softmax=True, squared_pred=True, to_onehot_y=True)
metric_function = monai.metrics.DiceMetric(include_background=False, reduction="mean")

val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()

for epoch in range(args.epochs):
    logger.info("-" * 10)
    logger.info(f"epoch {epoch + 1}/{args.epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for train_data in train_dataloader:
        step += 1
        train_inputs, train_labels = train_data["pet_ct"].to(device), train_data["mask"].to(device)
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        loss = loss_function(train_outputs, train_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_dataset) // train_dataloader.batch_size
        logger.info(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

    if step == 0:
        raise ValueError("step is 0, which means training data loader is empty or not working correctly.")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    scheduler.step(epoch_loss)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            val_step = 0
            for val_data in val_dataloader:
                val_inputs, val_labels = val_data["pet_ct"].to(device), val_data["mask"].to(device)
                val_outputs = model(val_inputs)
                dice_values = metric_function(val_outputs, val_labels)
                val_step += len(dice_values)
                metric_sum += dice_values.item() * len(dice_values)

            if val_step == 0:
                raise ValueError("val_step is 0, which means validation data loader is empty or not working correctly.")
            metric = metric_sum / val_step
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                save_path = os.path.join(args.save_dir, f"model_epoch_{best_metric_epoch}.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"New best model saved to {save_path}")

            logger.info(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )

logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
