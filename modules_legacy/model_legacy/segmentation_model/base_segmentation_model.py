import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSegmentationModel(nn.Module):

    def __init__(self):
        super().__init__()

        self._input_size = None


    @property
    def input_size(self):

        if self._input_size is None:
            raise RuntimeError(
                "input_size has not been configured."
            )

        return self._input_size


    def get_input_size(self):
        return self.input_size


    def get_feature_module(self):
        """
        Return the spatial semantic module whose
        output will be used for semantic guidance.

        Must return an nn.Module producing a
        [B, C, H, W] tensor.
        """

        raise NotImplementedError
    
    def freeze(self):

        self.eval()

        for param in self.parameters():
            param.requires_grad = False

        return self


    @property
    def num_classes(self):
        raise NotImplementedError