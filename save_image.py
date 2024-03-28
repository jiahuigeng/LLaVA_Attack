import pickle
import torch
import matplotlib.pyplot as plt
from PIL import Image
from llava.mm_utils import process_images
fact_bin = "facts_part_both_best_2954_prompt.bin"
fact_png = "facts_part_both_best_2954_prompt.png"

# # Assuming you have a (1, 3, 336, 336) torch tensor named `tensor`
# # For demonstration, let's create a dummy tensor with the same shape
# tensor = torch.tensor(pickle.load(open(fact_bin, "rb")))
#
# # Squeeze the tensor to remove the batch dimension, making it (3, 336, 336)
# image_tensor = tensor.squeeze(0)
# #
# # Convert the tensor to a numpy array and transpose it to (336, 336, 3) for plotting
# image_array = image_tensor.numpy().transpose(1, 2, 0)
#
# # Plot the image
# plt.imshow(image_array)
# plt.axis('off')  # Turn off axis numbers and ticks
#
# # Save the image as a PNG file
# plt.savefig(fact_png, bbox_inches='tight', pad_inches=0)
#
#
# image_tensor_from_png = Image.open(fact_png).convert('RGB')
# image_tensor_from_png = torch.tensor(image_tensor_from_png)# image_tensor.to(model.device, dtype=torch.float16)

import numpy as np
import imageio

# Example 3D array
original_array = np.random.rand(10, 10, 3)  # 10x10 array with 3 channels

# Save the 3D array to a PNG file
def save_array_to_png(array, file_name):
    imageio.imwrite(file_name, array)

# Load a PNG file back into a 3D NumPy array
def load_png_to_array(file_name, shape):
    # Load the image
    loaded_image = imageio.imread(file_name)
    # Reshape it back to its original shape
    reshaped_array = loaded_image.reshape(shape)
    return reshaped_array

# Save the original array
save_array_to_png(original_array, 'array_image.png')

# Load the array back from the PNG
loaded_array = load_png_to_array('array_image.png', original_array.shape)

# Check if the loaded array is the same as the original
print(np.allclose(original_array, loaded_array))
