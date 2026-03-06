import wandb
import numpy as np
from keras.datasets import mnist
from collections import defaultdict
from PIL import Image

PROJECT_NAME = 'DA6401_Assignment_1'

wandb.init(
    project=PROJECT_NAME,
    name="mnist_sample_images",
)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Create table with 5 images from each class
columns = ["Class", "Label", "Image"]
data = []

# Group images by class
class_indices = defaultdict(list)
for idx, label in enumerate(y_train):
    class_indices[label].append(idx)

# Sample 5 images from each of the 10 classes
for class_idx in range(10):
    available_indices = class_indices[class_idx]
    selected_indices = np.random.choice(
        available_indices, 
        size=5, 
        replace=False
    )
    
    for idx in selected_indices:
        img = x_train[idx]
        pil_img = Image.fromarray(img.astype(np.uint8))
        
        data.append([
            str(class_idx),
            class_idx,
            wandb.Image(pil_img)
        ])

# Log the table
table = wandb.Table(columns=columns, data=data)
wandb.log({"MNIST_Samples": table})

wandb.finish()
print("5 sample images from each MNIST class logged to W&B!")
