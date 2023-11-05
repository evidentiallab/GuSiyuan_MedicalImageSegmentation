import os.path
from glob import glob

import torch
import monai
from monai.data import Dataset, list_data_collate, decollate_batch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
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
from monai.visualize import plot_2d_or_3d_image
from monai.networks.nets import UNet

import argparse
import logging
import datetime
from torch.utils.tensorboard import SummaryWriter

# Configure logging with timestamp-based filename
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
# Check and create logs directory
if not os.path.exists("logs"):
    os.makedirs("logs")
log_filename = f"logs/training_{formatted_time}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_filename)])
logger = logging.getLogger()

parser = argparse.ArgumentParser(description="Training script for medical image segmentation")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument("--data_dir", type=str, default=r"C:\Users\lifel\Projects\Dataset", help="Directory containing the data")
parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save trained models")
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

train_session_dir = os.path.join(args.save_dir, f"unet_baseline_train_session_{formatted_time}")
os.makedirs(train_session_dir, exist_ok=True)

data_dir = args.data_dir
pet_dir = os.path.join(data_dir, 'SUV2')
ct_dir = os.path.join(data_dir, 'CTres2')
mask_dir = os.path.join(data_dir, 'SEG2')

pet_files = sorted(glob(os.path.join(pet_dir, '*SUV.nii.gz')))
ct_files = sorted(glob(os.path.join(ct_dir, '*CTres.nii.gz')))
mask_files = sorted(glob(os.path.join(mask_dir, '*SEG.nii.gz')))

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

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, collate_fn=list_data_collate, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0, collate_fn=list_data_collate, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(spatial_dims=3,
             in_channels=2,
             out_channels=2,
             kernel_size=5,
             channels=(8, 16, 32, 64, 128),
             strides=(2, 2, 2, 2),
             num_res_units=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
loss_function = monai.losses.DiceLoss(include_background=False, softmax=True, squared_pred=True, to_onehot_y=True)
metric_function = monai.metrics.DiceMetric(include_background=False, reduction="mean")
post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])

val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()

writer = SummaryWriter()

for epoch in range(args.epochs):
    batch_losses = []
    logger.info("-" * 10)
    logger.info(f"epoch {epoch + 1}/{args.epochs}")
    epoch_len = len(train_dataset) // train_dataloader.batch_size + (
            0 < len(train_dataset) % train_dataloader.batch_size)
    epoch_loss = 0
    step = 0
    model.train()
    for train_data in train_dataloader:
        step += 1
        train_inputs, train_labels = train_data["pet_ct"].to(device), train_data["mask"].to(device)
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        loss = loss_function(train_outputs, train_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        logger.info(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        batch_losses.append(loss.item())

    if step == 0:
        raise ValueError("step is 0, which means training data loader is empty or not working correctly.")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    writer.add_scalar('Loss/PerEpoch', epoch_loss, global_step=epoch)
    scheduler.step(epoch_loss)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            val_step = 0
            for val_data in val_dataloader:
                val_inputs, val_labels = val_data["pet_ct"].to(device), val_data["mask"].to(device)
                val_outputs = model(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                metric_function(y_pred=val_outputs, y=val_labels)
            metric = metric_function.aggregate().item()
            metric_function.reset()

            metric_values.append(metric)
            writer.add_scalar('Metric/PerEpoch', metric, global_step=epoch)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                save_path = os.path.join(train_session_dir, f"model_epoch_{best_metric_epoch}_{formatted_time}.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"New best model saved to {save_path}")

                for idx, loss_value in enumerate(batch_losses):
                    writer.add_scalar('Loss/BestMetricEpoch', loss_value, global_step=idx)
                writer.add_text('BestEpochNotes', f'Best metric epoch: {best_metric_epoch}, Metric: {best_metric:.4f}', global_step=0)

            logger.info(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )

            writer.add_scalar("val_mean_dice", metric, epoch + 1)

            plot_2d_or_3d_image(val_inputs, epoch + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

writer.close()
