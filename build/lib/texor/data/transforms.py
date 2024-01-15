from typing import Sequence, Callable, Union, Tuple
import numpy as np
from ..core import Tensor

class Transform:
    """Base class for all transforms"""
    def __call__(self, data: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        raise NotImplementedError
        
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

class Compose:
    """Composes several transforms together"""
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms
        
    def __call__(self, data: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        for transform in self.transforms:
            data = transform(data)
        return data
        
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '(['
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n])'
        return format_string

class ToTensor(Transform):
    """Convert ndarrays to Tensors"""
    def __call__(self, data: np.ndarray) -> Tensor:
        if isinstance(data, Tensor):
            return data
        return Tensor(np.asarray(data))

class Normalize(Transform):
    """Normalize a tensor image with mean and standard deviation"""
    def __init__(self, mean: Union[float, Sequence[float]], 
                 std: Union[float, Sequence[float]]):
        self.mean = np.array(mean)
        self.std = np.array(std)
        
    def __call__(self, tensor: Tensor) -> Tensor:
        if isinstance(tensor, np.ndarray):
            tensor = Tensor(tensor)
        return (tensor - self.mean) / self.std
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

class RandomHorizontalFlip(Transform):
    """Randomly flip the image horizontally"""
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() < self.p:
            if isinstance(tensor, np.ndarray):
                return np.flip(tensor, axis=-1)
            return Tensor(np.flip(tensor.numpy(), axis=-1))
        return tensor
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'

class RandomVerticalFlip(Transform):
    """Randomly flip the image vertically"""
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() < self.p:
            if isinstance(tensor, np.ndarray):
                return np.flip(tensor, axis=-2)
            return Tensor(np.flip(tensor.numpy(), axis=-2))
        return tensor
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'

class RandomRotation(Transform):
    """Rotate image by random angle"""
    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, (tuple, list)):
            self.min_angle = degrees[0]
            self.max_angle = degrees[1]
        else:
            self.min_angle = -degrees
            self.max_angle = degrees
            
    def __call__(self, tensor: Tensor) -> Tensor:
        angle = np.random.uniform(self.min_angle, self.max_angle)
        if isinstance(tensor, np.ndarray):
            from scipy.ndimage import rotate
            return rotate(tensor, angle, reshape=False)
        return Tensor(rotate(tensor.numpy(), angle, reshape=False))
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(degrees=({self.min_angle}, {self.max_angle}))'

class RandomCrop(Transform):
    """Crop image at a random location"""
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
            
    def __call__(self, tensor: Tensor) -> Tensor:
        if isinstance(tensor, np.ndarray):
            data = tensor
        else:
            data = tensor.numpy()
            
        h, w = data.shape[-2:]
        new_h, new_w = self.size
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        
        cropped = data[..., top:top+new_h, left:left+new_w]
        return Tensor(cropped) if isinstance(tensor, Tensor) else cropped
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'

class Resize(Transform):
    """Resize image to given size"""
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
            
    def __call__(self, tensor: Tensor) -> Tensor:
        if isinstance(tensor, np.ndarray):
            data = tensor
        else:
            data = tensor.numpy()
            
        from scipy.ndimage import zoom
        h, w = data.shape[-2:]
        scale_h, scale_w = self.size[0] / h, self.size[1] / w
        
        resized = zoom(data, (1,) * (data.ndim - 2) + (scale_h, scale_w))
        return Tensor(resized) if isinstance(tensor, Tensor) else resized
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'