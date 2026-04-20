from typing import Sequence, Callable, Union, Tuple, Optional
import numpy as np
from scipy.ndimage import rotate, zoom, affine_transform
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

        # Ensure crop size doesn't exceed image size
        new_h = min(new_h, h)
        new_w = min(new_w, w)

        top = np.random.randint(0, h - new_h + 1) if h > new_h else 0
        left = np.random.randint(0, w - new_w + 1) if w > new_w else 0

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

        h, w = data.shape[-2:]
        scale_h, scale_w = self.size[0] / h, self.size[1] / w

        resized = zoom(data, (1,) * (data.ndim - 2) + (scale_h, scale_w))
        return Tensor(resized) if isinstance(tensor, Tensor) else resized
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class RandomAffine(Transform):
    """Apply random affine transformation to image"""
    
    def __init__(self, degrees: Union[float, Tuple[float, float]] = 0,
                 translate: Optional[Tuple[float, float]] = None,
                 scale: Optional[Tuple[float, float]] = None,
                 shear: Optional[Union[float, Tuple[float, float]]] = None):
        self.degrees = degrees if isinstance(degrees, tuple) else (-degrees, degrees)
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, tensor: Tensor) -> Tensor:
        if isinstance(tensor, np.ndarray):
            data = tensor
        else:
            data = tensor.numpy()
        
        # Get random parameters
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        
        # Translation
        trans_x = 0
        trans_y = 0
        if self.translate is not None:
            trans_x = np.random.uniform(-self.translate[0], self.translate[0]) * data.shape[-1]
            trans_y = np.random.uniform(-self.translate[1], self.translate[1]) * data.shape[-2]
        
        # Scale
        scale_factor = 1.0
        if self.scale is not None:
            scale_factor = np.random.uniform(self.scale[0], self.scale[1])
        
        # Simple rotation (no full affine for simplicity)
        result = rotate(data, angle, reshape=False, mode='constant', cval=0)
        
        return Tensor(result) if isinstance(tensor, Tensor) else result

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(degrees={self.degrees}, translate={self.translate})'


class ColorJitter(Transform):
    """Randomly change brightness, contrast, saturation and hue"""
    
    def __init__(self, brightness: float = 0, contrast: float = 0,
                 saturation: float = 0, hue: float = 0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, tensor: Tensor) -> Tensor:
        if isinstance(tensor, np.ndarray):
            data = tensor.copy()
        else:
            data = tensor.numpy().copy()
        
        # Brightness
        if self.brightness > 0:
            factor = 1 + np.random.uniform(-self.brightness, self.brightness)
            data = data * factor
        
        # Contrast
        if self.contrast > 0:
            factor = 1 + np.random.uniform(-self.contrast, self.contrast)
            mean = data.mean(axis=(-2, -1), keepdims=True)
            data = (data - mean) * factor + mean
        
        # Saturation (for multi-channel images)
        if self.saturation > 0 and data.shape[-3] > 1:
            factor = 1 + np.random.uniform(-self.saturation, self.saturation)
            gray = data.mean(axis=-3, keepdims=True)
            data = gray + (data - gray) * factor
        
        return Tensor(data) if isinstance(tensor, Tensor) else data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(brightness={self.brightness}, contrast={self.contrast})'


class RandomPerspective(Transform):
    """Random perspective transformation"""
    
    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() > self.p:
            return tensor
        
        if isinstance(tensor, np.ndarray):
            data = tensor
        else:
            data = tensor.numpy()
        
        # Simplified perspective - just return original for now
        # Full implementation would use proper perspective transform
        return Tensor(data) if isinstance(tensor, Tensor) else data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(distortion_scale={self.distortion_scale})'


class RandomErasing(Transform):
    """Randomly erase a rectangular region of the image"""
    
    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3), value: float = 0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() > self.p:
            return tensor

        if isinstance(tensor, np.ndarray):
            data = tensor.copy()
        else:
            data = tensor.numpy().copy()

        h, w = data.shape[-2:]
        area = h * w

        for _ in range(10): # Try up to 10 times
            target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])

            erase_h = int(np.sqrt(target_area * aspect_ratio))
            erase_w = int(np.sqrt(target_area / aspect_ratio))

            if erase_h < h and erase_w < w:
                i = np.random.randint(0, h - erase_h)
                j = np.random.randint(0, w - erase_w)

                if data.ndim >= 3:
                    data[..., i:i+erase_h, j:j+erase_w] = self.value
                else:
                    data[i:i+erase_h, j:j+erase_w] = self.value
                break

        return Tensor(data) if isinstance(tensor, Tensor) else data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p}, scale={self.scale})'


class Pad(Transform):
    """Pad image to given size"""
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]],
                 fill: float = 0, padding_mode: str = 'constant'):
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, tensor: Tensor) -> Tensor:
        if isinstance(tensor, np.ndarray):
            data = tensor
        else:
            data = tensor.numpy()
        
        if self.padding_mode == 'constant':
            if data.ndim == 3:
                padded = np.pad(data, ((0, 0), (self.padding[0], self.padding[1]),
                                       (self.padding[2], self.padding[3])),
                               mode=self.padding_mode, constant_values=self.fill)
            else:
                padded = np.pad(data, ((self.padding[0], self.padding[1]),
                                       (self.padding[2], self.padding[3])),
                               mode=self.padding_mode, constant_values=self.fill)
        else:
            if data.ndim == 3:
                padded = np.pad(data, ((0, 0), (self.padding[0], self.padding[1]),
                                       (self.padding[2], self.padding[3])),
                               mode=self.padding_mode)
            else:
                padded = np.pad(data, ((self.padding[0], self.padding[1]),
                                       (self.padding[2], self.padding[3])),
                               mode=self.padding_mode)
        
        return Tensor(padded) if isinstance(tensor, Tensor) else padded

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(padding={self.padding})'


class CenterCrop(Transform):
    """Crop image at center to given size"""
    
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
        
        c, h, w = data.shape[-3], data.shape[-2], data.shape[-1]
        new_h, new_w = self.size
        
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        
        if data.ndim == 3:
            cropped = data[..., top:top+new_h, left:left+new_w]
        else:
            cropped = data[top:top+new_h, left:left+new_w]
        
        return Tensor(cropped) if isinstance(tensor, Tensor) else cropped

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class FiveCrop(Transform):
    """Crop image into four corners and center"""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, tensor: Tensor) -> Tuple[Tensor, ...]:
        if isinstance(tensor, np.ndarray):
            data = tensor
        else:
            data = tensor.numpy()
        
        c, h, w = data.shape[-3], data.shape[-2], data.shape[-1]
        new_h, new_w = self.size
        
        # Four corners
        crops = []
        positions = [
            (0, 0),  # top-left
            (0, w - new_w),  # top-right
            (h - new_h, 0),  # bottom-left
            (h - new_h, w - new_w),  # bottom-right
            ((h - new_h) // 2, (w - new_w) // 2)  # center
        ]
        
        for top, left in positions:
            if data.ndim == 3:
                crop = data[..., top:top+new_h, left:left+new_w]
            else:
                crop = data[top:top+new_h, left:left+new_w]
            crops.append(Tensor(crop) if isinstance(tensor, Tensor) else crop)
        
        return tuple(crops)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class TenCrop(Transform):
    """Crop image into four corners and center, plus horizontal flips"""
    
    def __init__(self, size: Union[int, Tuple[int, int]], vertical_flip: bool = False):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, tensor: Tensor) -> Tuple[Tensor, ...]:
        if isinstance(tensor, np.ndarray):
            data = tensor
        else:
            data = tensor.numpy()
        
        five_crop = FiveCrop(self.size)
        crops = list(five_crop(data))
        
        # Horizontal flip
        if data.ndim == 3:
            flipped = np.flip(data, axis=-1)
        else:
            flipped = np.flip(data, axis=-1)
        
        for crop in five_crop(flipped):
            crops.append(crop)
        
        return tuple(crops)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class RandomChoice(Transform):
    """Apply a random transform from a list"""
    
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms

    def __call__(self, tensor: Tensor) -> Tensor:
        transform = np.random.choice(self.transforms)
        return transform(tensor)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(transforms={self.transforms})'


class RandomApply(Transform):
    """Apply a transform with a given probability"""
    
    def __init__(self, transforms: Sequence[Transform], p: float = 0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() < self.p:
            for transform in self.transforms:
                tensor = transform(tensor)
        return tensor

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(transforms={self.transforms}, p={self.p})'


class Lambda(Transform):
    """Apply a lambda function as transform"""
    
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, data: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        return self.func(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(func={self.func})'


class ToPILImage(Transform):
    """Convert tensor to PIL Image"""
    
    def __init__(self, mode: Optional[str] = None):
        self.mode = mode

    def __call__(self, tensor: Tensor) -> 'PIL.Image.Image':
        from PIL import Image
        
        if isinstance(tensor, Tensor):
            data = tensor.numpy()
        else:
            data = tensor
        
        # Handle different dimensions
        if data.ndim == 3:
            # CHW -> HWC
            if data.shape[0] in [1, 3, 4]:
                data = np.transpose(data, (1, 2, 0))
        
        # Convert to uint8 if needed
        if data.dtype != np.uint8:
            if data.max() <= 1:
                data = (data * 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
        
        # Handle mode
        if self.mode == 'L' and data.shape[-1] == 3:
            data = np.mean(data, axis=-1).astype(np.uint8)
        elif self.mode == 'RGB' and data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)
        
        return Image.fromarray(data, mode=self.mode)


class ToNumpy(Transform):
    """Convert Tensor to numpy array"""
    
    def __call__(self, tensor: Tensor) -> np.ndarray:
        if isinstance(tensor, Tensor):
            return tensor.numpy()
        return tensor


class ToDtype(Transform):
    """Convert tensor to specified dtype"""
    
    def __init__(self, dtype: np.dtype):
        self.dtype = dtype

    def __call__(self, tensor: Tensor) -> Tensor:
        if isinstance(tensor, np.ndarray):
            return Tensor(tensor.astype(self.dtype))
        return Tensor(tensor.numpy().astype(self.dtype))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dtype={self.dtype})'


class RandomInvert(Transform):
    """Invert colors of an image"""
    
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() < self.p:
            if isinstance(tensor, np.ndarray):
                data = tensor
            else:
                data = tensor.numpy()
            
            if data.dtype == np.uint8:
                inverted = 255 - data
            else:
                inverted = 1 - data
            
            return Tensor(inverted) if isinstance(tensor, Tensor) else inverted
        return tensor

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomPosterize(Transform):
    """Reduce the number of bits for each color channel"""
    
    def __init__(self, bits: int, p: float = 0.5):
        self.bits = max(1, min(8, bits))
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() < self.p:
            if isinstance(tensor, np.ndarray):
                data = tensor
            else:
                data = tensor.numpy()
            
            # Calculate the number of levels
            levels = 2 ** self.bits
            # Scale to 0-255, quantize, then scale back
            quantized = (data * (levels - 1)).astype(np.uint8)
            result = quantized.astype(np.float32) / (levels - 1) if levels < 256 else data
            
            return Tensor(result) if isinstance(tensor, Tensor) else result
        return tensor

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bits={self.bits})'


class RandomSolarize(Transform):
    """Invert all pixels above a threshold"""
    
    def __init__(self, threshold: float = 0.5, p: float = 0.5):
        self.threshold = threshold
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() < self.p:
            if isinstance(tensor, np.ndarray):
                data = tensor
            else:
                data = tensor.numpy()
            
            mask = data > self.threshold
            inverted = 1 - data
            result = np.where(mask, inverted, data)
            
            return Tensor(result) if isinstance(tensor, Tensor) else result
        return tensor

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(threshold={self.threshold})'


class RandomAdjustSharpness(Transform):
    """Adjust sharpness of an image"""
    
    def __init__(self, sharpness_factor: float = 2.0, p: float = 0.5):
        self.sharpness_factor = sharpness_factor
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() < self.p:
            if isinstance(tensor, np.ndarray):
                data = tensor
            else:
                data = tensor.numpy()
            
            # Simple sharpness enhancement using unsharp masking
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(data, sigma=1)
            enhanced = data + (data - blurred) * (self.sharpness_factor - 1)
            
            return Tensor(enhanced) if isinstance(tensor, Tensor) else enhanced
        return tensor

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sharpness_factor={self.sharpness_factor})'


class RandomAutocontrast(Transform):
    """Autocontrast the image"""
    
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() < self.p:
            if isinstance(tensor, np.ndarray):
                data = tensor
            else:
                data = tensor.numpy()
            
            # For each channel, scale to 0-1 range
            if data.ndim == 3:
                for c in range(data.shape[0]):
                    channel = data[c]
                    min_val = channel.min()
                    max_val = channel.max()
                    if max_val > min_val:
                        data[c] = (channel - min_val) / (max_val - min_val)
            else:
                min_val = data.min()
                max_val = data.max()
                if max_val > min_val:
                    data = (data - min_val) / (max_val - min_val)
            
            return Tensor(data) if isinstance(tensor, Tensor) else data
        return tensor

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomEqualize(Transform):
    """Equalize the image histogram"""
    
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if np.random.random() < self.p:
            if isinstance(tensor, np.ndarray):
                data = tensor
            else:
                data = tensor.numpy()
            
            # Simple histogram equalization
            if data.ndim == 3:
                for c in range(data.shape[0]):
                    channel = (data[c] * 255).astype(np.uint8).flatten()
                    hist, bins = np.histogram(channel, 256, (0, 256))
                    cdf = hist.cumsum()
                    cdf_normalized = cdf / cdf[-1]
                    
                    # Interpolate
                    channel_equalized = np.interp(channel, bins[:-1], cdf_normalized)
                    data[c] = channel_equalized.reshape(data[c].shape)
            else:
                channel = (data * 255).astype(np.uint8).flatten()
                hist, bins = np.histogram(channel, 256, (0, 256))
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]
                
                channel_equalized = np.interp(channel, bins[:-1], cdf_normalized)
                data = channel_equalized.reshape(data.shape)
            
            return Tensor(data) if isinstance(tensor, Tensor) else data
        return tensor

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


# Utility functions
def get_transforms(name: str) -> Transform:
    """Get transform by name"""
    transforms = {
        'to_tensor': ToTensor,
        'normalize': Normalize,
        'random_horizontal_flip': RandomHorizontalFlip,
        'random_vertical_flip': RandomVerticalFlip,
        'random_rotation': RandomRotation,
        'random_crop': RandomCrop,
        'resize': Resize,
        'center_crop': CenterCrop,
        'pad': Pad,
        'color_jitter': ColorJitter,
        'random_erasing': RandomErasing,
    }
    
    name = name.lower()
    if name not in transforms:
        raise ValueError(f"Unknown transform: {name}")
    
    return transforms[name]()