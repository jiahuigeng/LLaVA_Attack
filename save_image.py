import pickle
import torch
import matplotlib.pyplot as plt

# Assuming you have a (1, 3, 336, 336) torch tensor named `tensor`
# For demonstration, let's create a dummy tensor with the same shape
tensor = torch.tensor(pickle.load(open("senti_part_both_best_2991_prompt.bin", "rb")))

# Squeeze the tensor to remove the batch dimension, making it (3, 336, 336)
image_tensor = tensor.squeeze(0)
#
# Convert the tensor to a numpy array and transpose it to (336, 336, 3) for plotting
image_array = image_tensor.numpy().transpose(1, 2, 0)

# Plot the image               
plt.imshow(image_array)
plt.axis('off')  # Turn off axis numbers and ticks

# Save the image as a PNG file
plt.savefig('senti_image.png', bbox_inches='tight', pad_inches=0)

