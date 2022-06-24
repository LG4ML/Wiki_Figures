from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def display_images(images: List[np.ndarray], titles: List[str], plot_shape: Tuple[int, int] = None, title: str = ''):
    if plot_shape is not None:
        fig, axs = plt.subplots(nrows=plot_shape[0], ncols=plot_shape[1], sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(nrows=(len(images)), sharex=True, sharey=True)
    axs = np.array(axs).flatten()
    fig.suptitle(title, fontsize=16)
    for index, image in enumerate(images):
        axs[index].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[index].set(title=titles[index])
    plt.tight_layout()
    plt.show()


# Load two consecutive frames
frame1 = cv2.imread(Path('../data/Images/Detection_1.JPG').__str__())
frame2 = cv2.imread(Path('../data/Images/Detection_2.JPG').__str__())

# Convert frames to grayscale
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Plot original images as well as grayscale images
display_images(images=[frame1, frame2, frame1_gray, frame2_gray], titles=['First Frame', 'Second Frame', '', ''],
               plot_shape=(2, 2))

# Calculate difference and apply threshold
abs_difference = cv2.absdiff(frame1_gray, frame2_gray)
_, binary = cv2.threshold(abs_difference, 50, 255, cv2.THRESH_BINARY)
display_images(images=[abs_difference, binary], titles=['Absolute Difference', 'Binary Image'], plot_shape=(1, 2))

# Apply dilation
dilated = cv2.dilate(binary, kernel=np.ones((3, 3), dtype=np.int8), iterations=15)
display_images(images=[binary, dilated], titles=['Binary Image', 'Dilated Image'], plot_shape=(1, 2))
