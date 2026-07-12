from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image
DATASET_ROOT = Path("datasets/raw/archive/train")

classes = sorted([folder for folder in DATASET_ROOT.iterdir() if folder.is_dir()])
fig,axes = plt.subplots(
    len(classes),
    3,
    figsize = (12,12)
)
fig.suptitle("Medvision data samples",fontsize = 18)
for row ,classes_folder in enumerate(classes):
    images = list(classes_folder.glob("*.jpg"))
    samples = random.sample(images,3)
    for col,image_path in enumerate(samples):
        image = Image.open(image_path)
        axes[row,col].imshow(image,cmap="grey")
        axes[row,col].set_title(classes_folder.name.capitalize())
        axes[row,col].axis("off")
plt.tight_layout()
plt.show()