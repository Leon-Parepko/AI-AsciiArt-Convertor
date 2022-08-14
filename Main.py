
"""
    This program could convert any image to
        it's corresponding Ascii-Art using
        convolutional neural network (as symbol
        classifier).
"""

# Import all libraries
import torch
import torchsummary


# Load image
img = ...

# Configure global variables
used_symbols = [...]
width, hight, depth = img.shape
img_shape = [width, hight, depth]
