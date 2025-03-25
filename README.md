# Neural Style Transfer with TensorFlow

This repository contains a complete implementation of Neural Style Transfer using TensorFlow 2 and a pretrained VGG19 network. The goal is to generate a new image that combines the content of one image with the style of another.

## Overview

Neural Style Transfer blends two images:
- **Content Image:** Provides the overall structure and layout.
- **Style Image:** Contributes colors, textures, and brushstroke patterns.

This is achieved by:
- Extracting intermediate feature representations from a pretrained VGG19 network.
- Defining a **content loss** (comparing content features) and a **style loss** (comparing Gram matrices of style features).
- Using gradient descent to iteratively update the input image so that it minimizes the total loss (weighted sum of content and style losses).

## Features

- **Image Download & Preparation:** Downloads content and style images from online sources and resizes them.
- **Preprocessing & Deprocessing:** Adjusts images to match VGG19 training conditions (BGR channels, channel-wise mean subtraction) and converts them back for display.
- **Model Construction:** Builds a custom model using the VGG19 architecture to extract desired intermediate features.
- **Loss Functions:** Implements both content loss (Euclidean distance between features) and style loss (using Gram matrices).
- **Optimization Loop:** Uses the Adam optimizer to iteratively update the input image (rather than model weights) to minimize loss.
- **Visualization:** Displays intermediate results and final outputs, with options to save the generated images.

## Requirements

- Python 3.x
- [TensorFlow 2.x](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow (PIL)](https://pillow.readthedocs.io/)
- (Optional) Google Colab for running in a notebook environment.

## Setup & Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/adityasingh0912/IMG_TO_ART.git
   cd neural-style-transfer
   ```

2. **Install Dependencies**

   You can install the required packages using pip:

   ```bash
   pip install tensorflow numpy matplotlib pillow
   ```

3. **Download the Notebook and Run**

   The code is provided in a Jupyter Notebook format. You can run it locally or upload it to [Google Colab](https://colab.research.google.com/drive/1XJz5B7xOSPSKe8wSDU3G9n2vdaMt2t3T#scrollTo=Qnz8HeXSXg6P/)
.

   If using Colab, simply open the notebook, and run the cells sequentially. The notebook downloads the necessary images, builds the model, and performs the style transfer.

## How It Works

1. **Image Preparation:**
   - Images are downloaded from the web.
   - They are resized with a maximum dimension of 512 pixels.
   - Preprocessing converts images to arrays, adjusts channels, and normalizes according to VGG19's requirements.

2. **Feature Extraction with VGG19:**
   - A pretrained VGG19 model is loaded.
   - Intermediate layers are selected to capture content (`block5_conv2`) and style (layers: `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`, `block5_conv1`).

3. **Defining Loss Functions:**
   - **Content Loss:** Mean squared error between the feature maps of the generated image and the content image.
   - **Style Loss:** Mean squared error between the Gram matrices (correlation of features) of the generated image and the style image.

4. **Optimization Process:**
   - The input image (initially the content image) is updated using gradient descent.
   - The Adam optimizer minimizes the combined loss (weighted sum of content and style losses).
   - Intermediate results are displayed, and the best output (lowest loss) is saved.

5. **Visualization & Output:**
   - Final images are deprocessed (reverse normalization and clipping) and displayed.
   - Code is provided to download the final images if run in Google Colab.

## Example Usage

To run style transfer on a pair of images:

```python
# Define content and style image paths
content_path = '/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg'
style_path = '/tmp/nst/The_Great_Wave_off_Kanagawa.jpg'

# Run the style transfer process
best_img, best_loss = run_style_transfer(content_path, style_path, num_iterations=1000)

# Display the results
show_results(best_img, content_path, style_path)
```

You can experiment with different image pairs by updating the image paths accordingly.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **VGG19 Model:** The style transfer implementation uses the pretrained VGG19 network available in TensorFlow.
- **Image Sources:**
  - [Green Sea Turtle](https://commons.wikimedia.org/wiki/File:Green_Sea_Turtle_grazing_seagrass.jpg)
  - [The Great Wave off Kanagawa](https://commons.wikimedia.org/wiki/File:The_Great_Wave_off_Kanagawa.jpg)
  - Other images are credited to their respective sources as indicated in the notebook.

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).
- [TensorFlow Neural Style Transfer Tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)
