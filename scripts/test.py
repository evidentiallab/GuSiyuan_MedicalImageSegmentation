import SimpleITK as sitk

# 加载图像
image = sitk.ReadImage(r"C:\Users\lifel\Projects\Evidential-neural-network-for-lymphoma-segmentation\Evidential_segmentation\LYMPHOMA\Data\CTres\PETCT_f9e0c504af_CTres.nii")

# 将SimpleITK图像转换为numpy数组
image_array = sitk.GetArrayFromImage(image)

# 找到最大和最小强度值
min_intensity = image_array.min()
max_intensity = image_array.max()

print(f"Minimum Intensity: {min_intensity}")
print(f"Maximum Intensity: {max_intensity}")
