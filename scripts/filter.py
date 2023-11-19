import SimpleITK as sitk
import os

mask_dir = '/path/to/input/directory'
empty_mask = []
empty_pet = []
empty_ct =[]

for mask_file in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, mask_file)
    if mask_path.endswith('.nii') or mask_path.endswith('nii.gz'):
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)

        if mask_array.sum() == 0:
            empty_mask.append(mask_path)
            empty_pet.append(mask_path.replace('SEG', 'SUV'))
            empty_ct.append(mask_path.replace('SEG', 'CTres'))

files_for_delete = empty_mask + empty_pet + empty_ct

for file in files_for_delete:
    try:
        os.remove(file)
    except OSError as e:
        print(f"Error deleting {file}: {e}")

files_for_delete = "\n".join(files_for_delete)

output_file = "croppeddata_deleted.txt"
with open(output_file, "w") as f:
    f.write(files_for_delete)

