import os.path

from model import *
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ConcatItemsd,
    ToTensord,
)
from monai.data import Dataset, list_data_collate
from glob import glob

data_dir = r"C:\Users\lifel\Projects\Evidential-neural-network-for-lymphoma-segmentation\Evidential_segmentation\LYMPHOMA\Data"
pet_dir = os.path.join(data_dir, 'SUV')
ct_dir = os.path.join(data_dir, 'CTres')
mask_dir = os.path.join(data_dir, 'SEG')

pet_files = sorted(glob(os.path.join(pet_dir, '*PET.nii.gz')))
ct_files = sorted(glob(os.path.join(ct_dir, '*CTres.nii.gz')))
mask_files = sorted(glob(os.path.join(mask_dir, '*SEG.nii.gz')))

data_dicts = [
    {"pet": pet_file, "ct": ct_file, "mask": mask_file}
    for pet_file, ct_file, mask_file in zip(pet_files, ct_files, mask_files)
]

transforms = Compose(
    [
        LoadImaged(keys=["pet", "ct"]),
        AddChanneld(keys=["pet", "ct"]),
        ConcatItemsd(keys=["pet", "ct"], name="pet_ct", dim=0),
        ToTensord(keys=["pet_ct"]),
    ]
)

dataset = Dataset(data=data_dicts, transform=transforms)

dataloader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=list_data_collate)