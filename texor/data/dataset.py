from typing import Sequence, Any, Callable, Optional, Iterator, Union, List, Tuple
import numpy as np
from ..core import Tensor
from ..core.native_backend import backend

class Dataset:
    """Base dataset class"""
    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform
        
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError
        
    def __len__(self) -> int:
        raise NotImplementedError
        
    def map(self, fn: Callable) -> 'Dataset':
        """Apply a function to each item in the dataset"""
        return MappedDataset(self, fn)

class TensorDataset(Dataset):
    """Dataset wrapping tensors"""
    def __init__(self, *tensors: Sequence[Union[Tensor, np.ndarray]], 
                 transform: Optional[Callable] = None):
        super().__init__(transform)
        
        # Convert numpy arrays to Tensors
        self.tensors = tuple(
            t if isinstance(t, Tensor) else Tensor(t)
            for t in tensors
        )
        
        # Validate lengths
        if not self.tensors:
            raise ValueError("At least one tensor required")
        if not all(len(t) == len(self.tensors[0]) for t in self.tensors):
            raise ValueError("All tensors must have the same length")
        
    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        items = tuple(tensor[index] for tensor in self.tensors)
        if self.transform:
            items = self.transform(items)
        return items
        
    def __len__(self) -> int:
        return len(self.tensors[0])

class DataLoader:
    """Data loader combining a dataset and a sampler"""
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory and backend.get_device() != 'cpu'
        
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if num_workers < 0:
            raise ValueError("num_workers must be greater than or equal to 0")
        
    def __iter__(self) -> Iterator:
        return DataLoaderIterator(self)
        
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class DataLoaderIterator:
    """Iterator for DataLoader"""
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.drop_last = loader.drop_last
        self.index = 0
        
        # Create indices for the entire dataset
        self.indices = np.arange(len(self.dataset))
        if loader.shuffle:
            np.random.shuffle(self.indices)
            
        # Initialize worker if needed
        if loader.num_workers > 0:
            self._init_workers()
            
    def __iter__(self) -> 'DataLoaderIterator':
        return self
        
    def __next__(self) -> Tuple[Tensor, ...]:
        if self.index >= len(self.dataset):
            raise StopIteration
            
        # Get indices for current batch
        end_idx = min(self.index + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.index:end_idx]
        self.index = end_idx
        
        # Check if we should drop last incomplete batch
        if len(batch_indices) < self.batch_size and self.drop_last:
            raise StopIteration
            
        # Get items for current batch
        try:
            batch = [self.dataset[i] for i in batch_indices]
        except Exception as e:
            raise RuntimeError(f"Error loading batch: {str(e)}")
        
        # Stack items into tensors
        try:
            result = tuple(map(lambda x: Tensor(np.stack(x)), zip(*batch)))
            
            # Move to device if needed
            if self.loader.pin_memory:
                result = tuple(t.to(backend.get_device()) for t in result)
                
            return result
        except Exception as e:
            raise RuntimeError(f"Error stacking batch: {str(e)}")
            
    def _init_workers(self):
        """Initialize worker processes for parallel loading"""
        # TODO: Implement multiprocessing for data loading
        pass

class ArrayDataset(Dataset):
    """Dataset wrapping numpy arrays"""
    def __init__(self, *arrays: Sequence[np.ndarray], 
                 transform: Optional[Callable] = None):
        super().__init__(transform)
        
        if not arrays:
            raise ValueError("At least one array required")
        if not all(len(arr) == len(arrays[0]) for arr in arrays):
            raise ValueError("All arrays must have the same length")
            
        self.arrays = arrays
        
    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        items = tuple(array[index] for array in self.arrays)
        if self.transform:
            items = self.transform(items)
        return items
        
    def __len__(self) -> int:
        return len(self.arrays[0])

class SubsetDataset(Dataset):
    """Subset of a dataset at specified indices"""
    def __init__(self, 
                 dataset: Dataset,
                 indices: Sequence[int],
                 transform: Optional[Callable] = None):
        super().__init__(transform)
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, idx: int) -> Any:
        if idx >= len(self):
            raise IndexError("Index out of range")
        return self.dataset[self.indices[idx]]
        
    def __len__(self) -> int:
        return len(self.indices)

class MappedDataset(Dataset):
    """Dataset that applies a function to another dataset"""
    def __init__(self, dataset: Dataset, fn: Callable):
        super().__init__()
        self.dataset = dataset
        self.fn = fn
        
    def __getitem__(self, idx: int) -> Any:
        item = self.dataset[idx]
        return self.fn(item)
        
    def __len__(self) -> int:
        return len(self.dataset)

def random_split(dataset: Dataset,
                lengths: Sequence[int],
                generator: Optional[np.random.Generator] = None) -> Tuple[SubsetDataset, ...]:
    """Randomly split a dataset into non-overlapping subsets"""
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of lengths must equal dataset length")
        
    indices = np.arange(len(dataset))
    if generator:
        generator.shuffle(indices)
    else:
        np.random.shuffle(indices)
        
    result = []
    offset = 0
    for length in lengths:
        result.append(SubsetDataset(dataset, indices[offset:offset + length]))
        offset += length
        
    return tuple(result)