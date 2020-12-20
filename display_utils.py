#-*- coding:utf-8 -*-
#!/etc/env python
'''
   @Author:Zhongxi Qiu
   @File: display_utils.py
   @Time: 2020-12-20 20:44:25
   @Version:1.0
'''
import torch


def denormalize(tensor:torch.Tensor, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    """Reverses the normalisation on a tensor.

    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.

    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean

    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor
        means (list or tuple): the mean of the normalized parameters
        stds (list or tupe): the std of the normalized parameters

    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)

    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]

    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized[0], means, stds):
        channel.mul_(std).add_(mean)

    return denormalized