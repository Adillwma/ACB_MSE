# ACB-MSE
Automatic-Class-Balanced MSE Loss function for PyTorch (ACB-MSE) to combat class imbalanced datasets. 

[![Build Status](https://img.shields.io/travis/username/repo.svg)](https://travis-ci.org/Adillwma/ACB-MSE)
[![Code Coverage](https://img.shields.io/codecov/c/github/username/repo.svg)](https://codecov.io/gh/Adillwma/ACB-MSE)
[![Language](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Introduction](#introduction)
- [Benefits](#benefits)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [License](#license)
- [Contributions](#contributions)
- [Contact](#contact)

## Introduction 
This repository contains the PyTorch implementation of the ACB-MSE loss function, which stands for Automatic Class Balanced Mean Squared Error, originally developed for the [DEEPCLEAN3D Denoiser](https://github.com/Adillwma/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector)

## Benefits
The ACB-MSE loss function was designed for data taken from particle detectors which often have a majority of 'pixels' which are unlit and a very sparse pattern of lit pixels. In this scenario the ACB-MSE loss provides two main benefits, addressing the class imbalance beteen lit and unlit pixels whilst also stabilising the loss gradient during training. Additonal parameters, 'A' and 'B', are provided to allow the user to set a custom balance between classes.

#### Variable Class Size - Training Stability
If the number of hit pixels varies dramatically between images during training it can cause the loss value to fluctuate. The ACB-MSE loss function solves this problem by automatically adjusting the weighting of the loss function to account for the frequency of each class in the target. 

<img src="Images/loss_curve_1.png" alt="Alternative Text" width="500">


The above plot demonstrates how each of the loss functions (ACB-MSE, MSE and MAE) behave based on the number of hits in the true signal. Two dummy images were created, the first image contains a simulated signal and the recovered image is created with 50% of that signal correctly identified, simulating a 50% signal recovery by the network. To generate the plot the first image was filled in two pixel increments with the second image following at a constant 50% recovery, and at each iteration the loss is calculated for the pair of images. We can see how the MSE and MAE functions loss varies as the size of the signal is increased with the recovery percentage fixed at 50%, whereas the ACB-MSE loss stays constant regardless of the frequency of the signal class.

#### Class Imbalance - Local Minima
Class imbalance is an issue that can arise where the interesting features are contained in the minority class. In the case of the DEEPCLEAN3D data the input images contained 11,264 total pixels, with only around 200 of them being hits. For the network, guessing that all the pixels are non-hits (zero valued) yields a very respectable reconstruction loss and is a simple transfer function for the network to learn, this local minima proved hard for the network to escape from. Class balancing based on class frequency is a simple solution to this problem that allows the loss value .


## Installation
Available on PyPi
```bash
pip install ACB-MSE
```

#### Requirements
- Python 3.x
- PyTorch (tested with version 2.0.1)



## Usage
#### Class Parameters
- `zero_weighting` (float, optional): A scalar weighting coefficient for the MSE loss of zero pixels in the target image. Default is 1.
- `nonzero_weighting` (float, optional): A scalar weighting coefficient for the MSE loss of non-zero pixels in the target image. Default is 1.

#### Inputs
   - Input (torch.Tensor): $(*)$, where $(*)$ means any number of dimensions.
   - Target (torch.Tensor): $(*)$, same shape as the input.

#### Returns
- Output (float): Calculated loss value.


##### Example Usage
```python
import torch
from acbmse import ACBMSE

# Select weighting for each class if not wanting to use the defualt 1:1 weighting
zero_weighting = 1.0
nonzero_weighting = 1.2

# Create an instance of the ACBMSE loss function with specified weighting coefficients
loss_function = ACBMSE(zero_weighting, nonzero_weighting)

# Dummy target image and reconstructed image tensors (assuming B=1, C=3, H=256, W=256)
target_image = torch.rand(1, 3, 256, 256)
reconstructed_image = torch.rand(1, 3, 256, 256)

# Calculate the ACBMSE loss
loss = loss_function(reconstructed_image, target_image)
print("ACB-MSE Loss:", loss.item())
```


## Methodology and Equations
1. It creates two masks from the target image:
   - `zero_mask`: A boolean mask where elements are `True` for zero-valued pixels in the target image.
   - `nonzero_mask`: A boolean mask where elements are `True` for non-zero-valued pixels in the target image.
2. Extracts the pixel values from both the target image and the reconstructed image corresponding to zero and non-zero masks.
3. Calculates the mean squared error loss for the pixel values associated with each mask.
4. Multiplies the losses with the corresponding weighting coefficients (`zero_weighting` and `nonzero_weighting`).
5. Returns the weighted MSE loss as the final loss value.

Please note that any NaN loss values resulting from the calculation are handled by setting them to zero.

The function relies on the knowledge of the indices for all hits and non-hits in the true label image, which are then compared to the values in the corresponding index's in the recovered image. The loss function is given by:

$$ \text{Loss} = A(\frac{1}{N _ h}\sum _ {i = 1} ^ {N _ h}(y _ i - \hat{y} _ i) ^ 2) + B(\frac{1}{N _ n}\sum _ {i = 1} ^ {N _ n}(y _ i - \hat{y} _ i) ^ 2) $$


where $y_i$ is the true value of the $i$-th pixel in the class, $\hat{y}_i$ is the predicted value of the $i$-th pixel in the class, and $n$ is the total number of pixels in the class (in our case labeled as $N_h$ and $N_n$ corresponding to 'hits' and 'no hits' classes, but can be extended to n classes). This approach to the loss function calculation takes the mean square of each class separately, when summing the separate classes errors back together they are automatically scaled by the inverse of the class frequency, normalising the class balance to 1:1. The additional coefficients $A$ and $B$ allow the user to manually adjust the balance to fine tune the balance.



## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contributions
Contributions to this codebase are welcome! If you encounter any issues or have suggestions for improvements please open an issue or a pull request on the [GitHub repository](https://github.com/Adillwma/ACB-MSE).

## Contact
For any inquiries, feel free to reach out to me at adillwmaa@gmail.com.















