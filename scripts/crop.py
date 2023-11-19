import os
import nibabel as nib
import numpy as np

def preprocess_nii_files(input_dir, output_dir, target_shape=(256, 256), slice_thickness=128):
    """
    Preprocess all NIfTI files in the given directory:
    - Center crop to 256x256 in width and height.
    - Pad along the depth (z-axis) to make it a multiple of 128.
    - Split into segments of 128 slices and save in the output directory.

    :param input_dir: Directory containing the NIfTI files.
    :param output_dir: Directory where processed files will be saved.
    :param target_shape: The target width and height after cropping (default: 256x256).
    :param slice_thickness: The thickness of each segment (default: 128).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, filename)
            # Load NIfTI file
            img = nib.load(file_path)
            data = img.get_fdata()
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Center crop
            center = [x // 2 for x in data.shape[:2]]
            cropped_data = data[
                center[0]-target_shape[0]//2:center[0]+target_shape[0]//2,
                center[1]-target_shape[1]//2:center[1]+target_shape[1]//2,
                :
            ]

            # Pad to make depth a multiple of slice_thickness
            pad_size = slice_thickness - cropped_data.shape[2] % slice_thickness
            if pad_size < slice_thickness:
                padded_data = np.pad(cropped_data, ((0, 0), (0, 0), (0, pad_size)), mode='constant')
            else:
                padded_data = cropped_data

            # Split and save segments
            num_segments = padded_data.shape[2] // slice_thickness
            for i in range(num_segments):
                segment = padded_data[:, :, i*slice_thickness:(i+1)*slice_thickness]
                segment_img = nib.Nifti1Image(segment, affine=np.eye(4))
                segment_filename = f"{i}_{filename}"
                nib.save(segment_img, os.path.join(output_dir, segment_filename))

# Example usage
input_directory = "/path/to/input/directory"  # Replace with your input directory path
output_directory = "/path/to/output/directory"  # Replace with your output directory path

preprocess_nii_files(input_directory, output_directory)
