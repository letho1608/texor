"""Data module for Texor

This module provides utilities for data loading, preprocessing,
and augmentation.
"""

from .dataset import (
    Dataset,
    TensorDataset,
    ArrayDataset,
    DataLoader,
    SubsetDataset,
    MappedDataset,
    random_split,
)

from .transforms import (
    # Base classes
    Transform,
    Compose,
    
    # Conversion transforms
    ToTensor,
    ToPILImage,
    ToNumpy,
    ToDtype,
    
    # Normalization transforms
    Normalize,
    
    # Geometric transforms
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomCrop,
    CenterCrop,
    FiveCrop,
    TenCrop,
    Pad,
    Resize,
    RandomAffine,
    RandomPerspective,
    
    # Color transforms
    ColorJitter,
    RandomInvert,
    RandomPosterize,
    RandomSolarize,
    RandomAdjustSharpness,
    RandomAutocontrast,
    RandomEqualize,
    
    # Advanced transforms
    RandomErasing,
    RandomChoice,
    RandomApply,
    Lambda,
    
    # Utility
    get_transforms,
)


__all__ = [
    # Dataset
    'Dataset',
    'TensorDataset',
    'ArrayDataset',
    'DataLoader',
    'SubsetDataset',
    'MappedDataset',
    'random_split',
    
    # Transforms
    'Transform',
    'Compose',
    'ToTensor',
    'ToPILImage',
    'ToNumpy',
    'ToDtype',
    'Normalize',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'RandomCrop',
    'CenterCrop',
    'FiveCrop',
    'TenCrop',
    'Pad',
    'Resize',
    'RandomAffine',
    'RandomPerspective',
    'ColorJitter',
    'RandomInvert',
    'RandomPosterize',
    'RandomSolarize',
    'RandomAdjustSharpness',
    'RandomAutocontrast',
    'RandomEqualize',
    'RandomErasing',
    'RandomChoice',
    'RandomApply',
    'Lambda',
    'get_transforms',
]