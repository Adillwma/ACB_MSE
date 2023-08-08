import torch

class acb_mse(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1):
        """
        Initializes the ACB-MSE Loss Function class with weighting coefficients.

        Args:
        - zero_weighting: a scalar weighting coefficient for the MSE loss of zero pixels
        - nonzero_weighting: a scalar weighting coefficient for the MSE loss of non-zero pixels
        """
        super().__init__()   
        self.zero_weighting = zero_weighting
        self.nonzero_weighting = nonzero_weighting
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, reconstructed_image, target_image):
        """
        Calculates the weighted mean squared error (MSE) loss between target_image and reconstructed_image.
        The loss for zero pixels in the target_image is weighted by zero_weighting, and the loss for non-zero
        pixels is weighted by nonzero_weighting.

        Args:
        - target_image: a tensor of shape (B, C, H, W) containing the target image
        - reconstructed_image: a tensor of shape (B, C, H, W) containing the reconstructed image

        Returns:
        - weighted_mse_loss: a scalar tensor containing the weighted MSE loss
        """
        zero_mask = (target_image == 0)
        nonzero_mask = ~zero_mask

        values_zero = target_image[zero_mask]
        values_nonzero = target_image[nonzero_mask]

        corresponding_values_zero = reconstructed_image[zero_mask]
        corresponding_values_nonzero = reconstructed_image[nonzero_mask]

        zero_loss = self.mse_loss(corresponding_values_zero, values_zero)
        nonzero_loss = self.mse_loss(corresponding_values_nonzero, values_nonzero)

        if torch.isnan(zero_loss):
            zero_loss = 0
        if torch.isnan(nonzero_loss):
            nonzero_loss = 0

        weighted_mse_loss = (self.zero_weighting * zero_loss) + (self.nonzero_weighting * nonzero_loss)

        return float(weighted_mse_loss)



# Select weighting for each class if not wanting to use the defualt 1:1 weighting
zero_weighting = 1.0
nonzero_weighting = 1.2

# Create an instance of the ACBMSE loss function with specified weighting coefficients
loss_function = acb_mse(zero_weighting, nonzero_weighting)

# Dummy target image and reconstructed image tensors (assuming B=10, C=3, H=256, W=256)
target_image = torch.rand(10, 3, 256, 256)
reconstructed_image = torch.rand(10, 3, 256, 256)

# Calculate the ACBMSE loss
loss = loss_function(reconstructed_image, target_image)
print("ACB-MSE Loss:", loss)