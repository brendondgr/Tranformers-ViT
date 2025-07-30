"""
Dataset utilities for Vision Transformer implementation.
Provides MNIST data loading, preprocessing, and validation functionality.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
from typing import Tuple, Optional


class MNISTDataLoader:
    """
    MNIST dataset loader with preprocessing and validation.
    
    Handles MNIST dataset loading with torchvision, applies appropriate
    preprocessing and normalization, and creates train/test data loaders
    with proper batching.
    """
    
    def __init__(self, 
                 img_size: int = 28,
                 in_channels: int = 1,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 download: bool = True,
                 data_root: str = './data'):
        """
        Initialize MNIST data loader.
        
        Args:
            img_size: Expected image size (assumes square images)
            in_channels: Expected number of input channels
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            download: Whether to download MNIST if not present
            data_root: Root directory for dataset storage
        """
        self.img_size = img_size
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.data_root = data_root
        
        # Validate parameters
        self._validate_parameters()
        
        # Create transforms
        self.transform = self._create_transforms()
        
        logging.info(f"MNISTDataLoader initialized with img_size={img_size}, "
                    f"in_channels={in_channels}, batch_size={batch_size}")
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        
        if self.in_channels not in [1, 3]:
            raise ValueError(f"in_channels must be 1 or 3 for MNIST, got {self.in_channels}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        # MNIST is 28x28, warn if different size is requested
        if self.img_size != 28:
            logging.warning(f"MNIST images are 28x28, but img_size={self.img_size} requested. "
                          f"Images will be resized.")
    
    def _create_transforms(self) -> transforms.Compose:
        """
        Create preprocessing transforms for MNIST data.
        
        Returns:
            Composed transforms for preprocessing
        """
        transform_list = []
        
        # Resize if needed (MNIST is natively 28x28)
        if self.img_size != 28:
            transform_list.append(transforms.Resize((self.img_size, self.img_size)))
        
        # Convert PIL Image to tensor
        transform_list.append(transforms.ToTensor())
        
        # Convert grayscale to RGB if needed
        if self.in_channels == 3:
            # Repeat grayscale channel 3 times to create RGB
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
        # Normalize to [0, 1] range (ToTensor already does this)
        # MNIST pixel values are in [0, 255], ToTensor converts to [0, 1]
        # Additional normalization can be added here if needed
        
        return transforms.Compose(transform_list)
    
    def load_datasets(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Load MNIST train and test datasets.
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        
        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            # Load training dataset
            train_dataset = torchvision.datasets.MNIST(
                root=self.data_root,
                train=True,
                transform=self.transform,
                download=self.download
            )
            
            # Load test dataset
            test_dataset = torchvision.datasets.MNIST(
                root=self.data_root,
                train=False,
                transform=self.transform,
                download=self.download
            )
            
            logging.info(f"MNIST datasets loaded: {len(train_dataset)} train samples, "
                        f"{len(test_dataset)} test samples")
            
            return train_dataset, test_dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MNIST datasets: {str(e)}")
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test data loaders with proper batching.
        
        Returns:
            Tuple of (train_loader, test_loader)
        
        Raises:
            RuntimeError: If data loader creation fails
        """
        try:
            # Load datasets
            train_dataset, test_dataset = self.load_datasets()
            
            # Create train data loader with shuffling
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()  # Pin memory if CUDA available
            )
            
            # Create test data loader without shuffling
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            logging.info(f"Data loaders created: {len(train_loader)} train batches, "
                        f"{len(test_loader)} test batches")
            
            return train_loader, test_loader
            
        except Exception as e:
            raise RuntimeError(f"Failed to create data loaders: {str(e)}")
    
    def validate_data_compatibility(self, sample_batch: torch.Tensor) -> bool:
        """
        Validate that loaded data matches expected dimensions.
        
        Args:
            sample_batch: Sample batch tensor to validate
            
        Returns:
            True if data is compatible
            
        Raises:
            ValueError: If data dimensions don't match expectations
        """
        if len(sample_batch.shape) != 4:
            raise ValueError(f"Expected 4D tensor (batch, channels, height, width), "
                           f"got shape {sample_batch.shape}")
        
        batch_size, channels, height, width = sample_batch.shape
        
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}")
        
        if height != self.img_size or width != self.img_size:
            raise ValueError(f"Expected image size {self.img_size}x{self.img_size}, "
                           f"got {height}x{width}")
        
        logging.info(f"Data compatibility validated: shape {sample_batch.shape}")
        return True
    
    def get_sample_batch(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample batch from data loader for testing.
        
        Args:
            data_loader: Data loader to sample from
            
        Returns:
            Tuple of (images, labels) tensors
        """
        try:
            data_iter = iter(data_loader)
            images, labels = next(data_iter)
            
            logging.info(f"Sample batch obtained: images shape {images.shape}, "
                        f"labels shape {labels.shape}")
            
            return images, labels
            
        except Exception as e:
            raise RuntimeError(f"Failed to get sample batch: {str(e)}")


def create_mnist_loaders(img_size: int = 28,
                        in_channels: int = 1,
                        batch_size: int = 32,
                        num_workers: int = 0,
                        download: bool = True,
                        data_root: str = './data') -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to create MNIST data loaders.
    
    Args:
        img_size: Expected image size (assumes square images)
        in_channels: Expected number of input channels (1 or 3)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        download: Whether to download MNIST if not present
        data_root: Root directory for dataset storage
        
    Returns:
        Tuple of (train_loader, test_loader)
        
    Raises:
        RuntimeError: If data loader creation fails
    """
    mnist_loader = MNISTDataLoader(
        img_size=img_size,
        in_channels=in_channels,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        data_root=data_root
    )
    
    return mnist_loader.create_data_loaders()


def validate_mnist_compatibility(train_loader: DataLoader,
                                test_loader: DataLoader,
                                expected_img_size: int,
                                expected_channels: int) -> bool:
    """
    Validate MNIST data loader compatibility with model requirements.
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        expected_img_size: Expected image size
        expected_channels: Expected number of channels
        
    Returns:
        True if compatible
        
    Raises:
        ValueError: If data is incompatible
    """
    # Create temporary loader instance for validation
    validator = MNISTDataLoader(
        img_size=expected_img_size,
        in_channels=expected_channels
    )
    
    # Validate train loader
    train_images, train_labels = validator.get_sample_batch(train_loader)
    validator.validate_data_compatibility(train_images)
    
    # Validate test loader
    test_images, test_labels = validator.get_sample_batch(test_loader)
    validator.validate_data_compatibility(test_images)
    
    logging.info("MNIST data loader compatibility validation successful")
    return True


def test_dataset_with_various_sizes(img_sizes: list = None,
                                   channel_configs: list = None,
                                   batch_size: int = 32) -> dict:
    """
    Test MNIST dataset loading with various image sizes and channel configurations.
    
    Args:
        img_sizes: List of image sizes to test (default: [28, 32, 64])
        channel_configs: List of channel configurations to test (default: [1, 3])
        batch_size: Batch size for testing
        
    Returns:
        Dictionary with test results for each configuration
    """
    if img_sizes is None:
        img_sizes = [28, 32, 64]
    
    if channel_configs is None:
        channel_configs = [1, 3]
    
    results = {}
    
    for img_size in img_sizes:
        for channels in channel_configs:
            config_name = f"size_{img_size}_channels_{channels}"
            
            try:
                # Create data loader with specific configuration
                mnist_loader = MNISTDataLoader(
                    img_size=img_size,
                    in_channels=channels,
                    batch_size=batch_size,
                    download=False  # Don't re-download for each test
                )
                
                train_loader, test_loader = mnist_loader.create_data_loaders()
                
                # Get sample batches
                train_images, train_labels = mnist_loader.get_sample_batch(train_loader)
                test_images, test_labels = mnist_loader.get_sample_batch(test_loader)
                
                # Validate compatibility
                mnist_loader.validate_data_compatibility(train_images)
                mnist_loader.validate_data_compatibility(test_images)
                
                results[config_name] = {
                    "status": "success",
                    "train_shape": train_images.shape,
                    "test_shape": test_images.shape,
                    "train_batches": len(train_loader),
                    "test_batches": len(test_loader),
                    "img_size": img_size,
                    "channels": channels
                }
                
                logging.info(f"Configuration {config_name} validated successfully")
                
            except Exception as e:
                results[config_name] = {
                    "status": "failed",
                    "error": str(e),
                    "img_size": img_size,
                    "channels": channels
                }
                
                logging.error(f"Configuration {config_name} failed: {str(e)}")
    
    return results


def validate_data_loader_properties(data_loader: DataLoader,
                                   expected_batch_size: int,
                                   expected_num_batches: Optional[int] = None) -> bool:
    """
    Validate data loader properties and consistency.
    
    Args:
        data_loader: Data loader to validate
        expected_batch_size: Expected batch size
        expected_num_batches: Expected number of batches (optional)
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Check if data loader is not empty
        if len(data_loader) == 0:
            raise ValueError("Data loader is empty")
        
        # Validate expected number of batches
        if expected_num_batches is not None:
            if len(data_loader) != expected_num_batches:
                raise ValueError(f"Expected {expected_num_batches} batches, got {len(data_loader)}")
        
        # Sample multiple batches to check consistency
        batch_count = 0
        total_samples = 0
        
        for batch_images, batch_labels in data_loader:
            batch_count += 1
            current_batch_size = batch_images.shape[0]
            total_samples += current_batch_size
            
            # Check batch size (last batch might be smaller)
            if batch_count < len(data_loader):  # Not the last batch
                if current_batch_size != expected_batch_size:
                    raise ValueError(f"Expected batch size {expected_batch_size}, "
                                   f"got {current_batch_size} in batch {batch_count}")
            else:  # Last batch can be smaller
                if current_batch_size > expected_batch_size:
                    raise ValueError(f"Last batch size {current_batch_size} exceeds "
                                   f"expected batch size {expected_batch_size}")
            
            # Validate tensor properties
            if len(batch_images.shape) != 4:
                raise ValueError(f"Expected 4D image tensor, got {len(batch_images.shape)}D")
            
            if len(batch_labels.shape) != 1:
                raise ValueError(f"Expected 1D label tensor, got {len(batch_labels.shape)}D")
            
            if batch_images.shape[0] != batch_labels.shape[0]:
                raise ValueError(f"Batch size mismatch: images {batch_images.shape[0]}, "
                               f"labels {batch_labels.shape[0]}")
            
            # Check for reasonable value ranges
            if batch_images.min() < 0 or batch_images.max() > 1:
                logging.warning(f"Image values outside [0,1] range: "
                              f"min={batch_images.min():.3f}, max={batch_images.max():.3f}")
            
            # Only check first few batches for efficiency
            if batch_count >= 3:
                break
        
        logging.info(f"Data loader validation successful: {batch_count} batches checked, "
                    f"{total_samples} samples processed")
        
        return True
        
    except Exception as e:
        raise ValueError(f"Data loader validation failed: {str(e)}")


class DatasetCompatibilityValidator:
    """
    Comprehensive dataset compatibility validator for Vision Transformer models.
    """
    
    def __init__(self, model_config: dict):
        """
        Initialize validator with model configuration.
        
        Args:
            model_config: Dictionary containing model configuration parameters
        """
        self.img_size = model_config.get('img_size', 28)
        self.in_channels = model_config.get('in_channels', 1)
        self.num_classes = model_config.get('num_classes', 10)
        self.patch_size = model_config.get('patch_size', 7)
        
    def validate_full_compatibility(self, train_loader: DataLoader, test_loader: DataLoader) -> dict:
        """
        Perform comprehensive compatibility validation.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            
        Returns:
            Dictionary with detailed validation results
        """
        results = {
            "overall_status": "unknown",
            "train_validation": {},
            "test_validation": {},
            "cross_validation": {},
            "errors": []
        }
        
        try:
            # Validate individual loaders
            results["train_validation"] = self._validate_single_loader(train_loader, "train")
            results["test_validation"] = self._validate_single_loader(test_loader, "test")
            
            # Cross-validation between loaders
            results["cross_validation"] = self._validate_cross_compatibility(train_loader, test_loader)
            
            # Check if all validations passed
            all_passed = (
                results["train_validation"]["status"] == "success" and
                results["test_validation"]["status"] == "success" and
                results["cross_validation"]["status"] == "success"
            )
            
            results["overall_status"] = "success" if all_passed else "failed"
            
        except Exception as e:
            results["overall_status"] = "error"
            results["errors"].append(str(e))
        
        return results
    
    def _validate_single_loader(self, data_loader: DataLoader, loader_name: str) -> dict:
        """Validate a single data loader."""
        try:
            # Get sample batch
            data_iter = iter(data_loader)
            images, labels = next(data_iter)
            
            # Validate dimensions
            if images.shape[1] != self.in_channels:
                raise ValueError(f"Channel mismatch: expected {self.in_channels}, got {images.shape[1]}")
            
            if images.shape[2] != self.img_size or images.shape[3] != self.img_size:
                raise ValueError(f"Size mismatch: expected {self.img_size}x{self.img_size}, "
                               f"got {images.shape[2]}x{images.shape[3]}")
            
            # Validate patch compatibility
            if self.img_size % self.patch_size != 0:
                raise ValueError(f"Image size {self.img_size} not divisible by patch size {self.patch_size}")
            
            return {
                "status": "success",
                "batch_shape": images.shape,
                "label_shape": labels.shape,
                "num_batches": len(data_loader)
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _validate_cross_compatibility(self, train_loader: DataLoader, test_loader: DataLoader) -> dict:
        """Validate compatibility between train and test loaders."""
        try:
            # Get sample batches
            train_images, train_labels = next(iter(train_loader))
            test_images, test_labels = next(iter(test_loader))
            
            # Check shape consistency
            if train_images.shape[1:] != test_images.shape[1:]:
                raise ValueError(f"Shape mismatch between train {train_images.shape[1:]} "
                               f"and test {test_images.shape[1:]}")
            
            # Check label ranges
            train_label_range = (train_labels.min().item(), train_labels.max().item())
            test_label_range = (test_labels.min().item(), test_labels.max().item())
            
            if train_label_range[1] >= self.num_classes or test_label_range[1] >= self.num_classes:
                raise ValueError(f"Label range exceeds num_classes {self.num_classes}")
            
            return {
                "status": "success",
                "train_label_range": train_label_range,
                "test_label_range": test_label_range
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }