# ACB-MSE
Automatic-Class-Balanced MSE Loss Method for PyTorch (ACB-MSE) to combat class imbalanced datasets. 

[![Build Status](https://img.shields.io/travis/username/repo.svg)](https://travis-ci.org/username/repo)
[![Code Coverage](https://img.shields.io/codecov/c/github/username/repo.svg)](https://codecov.io/gh/username/repo)
[![Language](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Documentation](#documentation)
- [License](#license)
- [Contributions](#contributions)
- [Contact](#contact)
- [References](#references)



## Introduction 
During development when tested with popular loss functions such as MSE and MAE the autoencoder was able to recover the signal, if the data contained many images of only the one signal pattern, which as mentioned results in a very poorly generalisable model is not usefull, but did indicate to us that the network is capable of encoding the pattern. However once shifting and scaling of the cross pattern were introduced it was unable to learn it and began returning blank images. After some research and investigation the problem was found to be an effect known as class imbalance which is an issue that can arise where the interesting features are contained in the minority class. The input image contains 11,264 total pixels and around 230 of them are hits (signal and noise) leaving 98.2\% of pixels as non-hits. For the network, just guessing that all the pixels are non-hits yields a 98.2\% reconstruction loss and it can easily get stuck in this local minima.

Class imbalance most commonly appears in classification tasks such as recognising that an image is of a certain class, i.e. ‘cat’, ‘dog', etc. Classification networks often use cross entropy loss and there are specific modifications of it developed to combat class imbalance, such as focal loss \cite{lin2017focal}, Lovász-Softmax\cite{berman2018lovasz} and class-balanced loss \cite{cui2019class}. We present a new similar method called 'Automatic Class Balanced MSE' (ACB-MSE) which instead, through simple modification of the MSE loss function, provides automatic class balancing for MSE loss with additional user adjustable weighting to further tune the networks response. 


## Installation


## Usage




## Examples
![Example Image](Images\loss_curve_1.png "Alternative Text")

Figure that demonstrates how each of the loss functions (ACB-MSE, MSE and MAE) behave based on the number of hits in the true signal. Two dummy images were created, the first image contains some ToF values of 100 the second image is a replica of the first but only containing the Tof values in half of the number of pixels of the first image, this simulates a 50\% signal recovery. to generate the plot the first image was filled in two pixel increments with the second image following at a constant 50\% recovery, and at each iteration the loss functions are calculated for the pair of images. We can see how the MSE and MAE functions loss varies as the size of the signal is increased. Whereas the ACB-MSE loss stays constant regardless of the frequency of the signal class.

The Loss functions response curve is demonstrated in fig \ref{fig:losscurves}. This show how the the ACB-MSE compares to vanilla MSE and also MAE. The addition of ACB balancing means that the separate classes (non hits and hits) are exactly balanced regardless of how large a proportion of the population they are. this means that by guessing all the pixels are non hits results in a loss of 50\% rather than 98.2\%, and to improve on this the network must begin to fill in signal pixels. This frees the network from the local minima of returning blank images and incentives it to put forward its best signal prediction. 

![Example Image](Images\loss_curve_2.png "Alternative Text")

Explain image 2


## Documentation

Link to comprehensive documentation or create a separate "Documentation" section within the README. Include in-depth explanations of the methodology, its mathematical foundation, and any underlying principles. Document key classes, functions, or modules and provide usage examples and API references. Consider including tutorials or guides to help users dive deeper into the methodology.

The function relies on the knowledge of the indices for all hits and non-hits in the true label image, which are then compared to the values in the corresponding index's in the recovered image. The loss function is given by:


$$\text{Loss} = A(\frac{1}{N_h}\sum_{i=1}^{N_h}(y_i - \hat{y}_i)^2) + B(\frac{1}{N_n}\sum_{i=1}^{N_n}(y_i - \hat{y}_i)^2) $$


where $y_i$ is the true value of the $i$-th pixel in the class, $\hat{y}_i$ is the predicted value of the $i$-th pixel in the class, and $n$ is the total number of pixels in the class (in our case labeled as $N_h$ and $N_n$ corresponding to 'hits' and 'no hits' classes, but can be extended to n classes). This approach to the loss function calculation takes the mean square of each class separately, when summing the separate classes errors back together they are automatically scaled by the inverse of the class frequency, normalising the class balance to 1:1. The additional coefficients $A$ and $B$ allow the user to manually adjust the balance to fine tune the networks results.


## License
Form-Factor Calculator is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributions
Contributions to this codebase are welcome! If you encounter any issues or have suggestions for improvements please open an issue or a pull request on the [GitHub repository](https://github.com/Adillwma/ACB-MSE).

## Contact
For any inquiries, feel free to reach out to me at adillwmaa@gmail.com.


## References:
[ACB-MSE](my paper)

[Automatic-Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)

[Class Imbalence](\cite{chawla2002smote})

[Focal Loss](\cite{lin2017focal})

[Lovász-Softmax](\cite{berman2018lovasz})

[Class Balanced Loss](\cite{cui2019class})