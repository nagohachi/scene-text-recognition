from abc import ABCMeta

import torch.nn as nn


class FeatureExtractorBase(nn.Module, metaclass=ABCMeta):
    pass
