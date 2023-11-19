import SimpleITK as sitk

# load the NIfTI file
image = sitk.ReadImage("/path/to/input/directory")

# use SimpleITK transfer to numpy array
image_array = sitk.GetArrayFromImage(image)

# find the max and min number
min_intensity = image_array.min()
max_intensity = image_array.max()

print(f"Minimum Intensity: {min_intensity}")
print(f"Maximum Intensity: {max_intensity}")
print(image_array.shape)
