"""
Configuration validation utilities for Vision Transformer implementation.
Provides type conversion, validation, and error handling for config.ini parameters.
"""

import configparser
import logging
from typing import Dict, Any, Union


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""

    pass


class ConfigurationManager:
    """Manages configuration loading, validation, and parameter extraction."""

    def __init__(self, config_path: str = "config.ini"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file with error handling."""
        try:
            self.config.read(self.config_path)
            if not self.config.sections():
                raise ConfigurationError(
                    f"Configuration file '{self.config_path}' is empty or not found"
                )
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")

    def _get_int_value(
        self, section: str, key: str, min_val: int = None, max_val: int = None
    ) -> int:
        """
        Get integer value with validation.

        Args:
            section: Configuration section name
            key: Configuration key
            min_val: Minimum allowed value (optional)
            max_val: Maximum allowed value (optional)

        Returns:
            Validated integer value

        Raises:
            ConfigurationError: If value is invalid or out of range
        """
        try:
            value = self.config.getint(section, key)
            if min_val is not None and value < min_val:
                raise ConfigurationError(
                    f"{section}.{key} must be >= {min_val}, got {value}"
                )
            if max_val is not None and value > max_val:
                raise ConfigurationError(
                    f"{section}.{key} must be <= {max_val}, got {value}"
                )
            return value
        except ValueError:
            raw_value = self.config.get(section, key, fallback="<missing>")
            raise ConfigurationError(
                f"{section}.{key} must be an integer, got '{raw_value}'"
            )
        except configparser.NoSectionError:
            raise ConfigurationError(f"Configuration section '[{section}]' not found")
        except configparser.NoOptionError:
            raise ConfigurationError(
                f"Configuration key '{key}' not found in section '[{section}]'"
            )

    def _get_string_value(
        self, section: str, key: str, allowed_values: list = None
    ) -> str:
        """
        Get string value with validation.

        Args:
            section: Configuration section name
            key: Configuration key
            allowed_values: List of allowed values (optional)

        Returns:
            Validated string value

        Raises:
            ConfigurationError: If value is invalid
        """
        try:
            value = self.config.get(section, key)
            if allowed_values and value not in allowed_values:
                raise ConfigurationError(
                    f"{section}.{key} must be one of {allowed_values}, got '{value}'"
                )
            return value
        except configparser.NoSectionError:
            raise ConfigurationError(f"Configuration section '[{section}]' not found")
        except configparser.NoOptionError:
            raise ConfigurationError(
                f"Configuration key '{key}' not found in section '[{section}]'"
            )

    def _get_float_value(
        self, section: str, key: str, min_val: float = None, max_val: float = None
    ) -> float:
        """
        Get float value with validation.

        Args:
            section: Configuration section name
            key: Configuration key
            min_val: Minimum allowed value (optional)
            max_val: Maximum allowed value (optional)

        Returns:
            Validated float value

        Raises:
            ConfigurationError: If value is invalid or out of range
        """
        try:
            value = self.config.getfloat(section, key)
            if min_val is not None and value < min_val:
                raise ConfigurationError(
                    f"{section}.{key} must be >= {min_val}, got {value}"
                )
            if max_val is not None and value > max_val:
                raise ConfigurationError(
                    f"{section}.{key} must be <= {max_val}, got {value}"
                )
            return value
        except ValueError:
            raw_value = self.config.get(section, key, fallback="<missing>")
            raise ConfigurationError(
                f"{section}.{key} must be a float, got '{raw_value}'"
            )
        except configparser.NoSectionError:
            raise ConfigurationError(f"Configuration section '[{section}]' not found")
        except configparser.NoOptionError:
            raise ConfigurationError(
                f"Configuration key '{key}' not found in section '[{section}]'"
            )

    def get_general_config(self) -> Dict[str, str]:
        """
        Get general configuration parameters.

        Returns:
            Dictionary containing general configuration

        Raises:
            ConfigurationError: If required parameters are missing or invalid
        """
        try:
            return {
                "model_name": self._get_string_value(
                    "General", "model_name", ["VisionTransformer", "BasicTransformer"]
                ),
                "dataset_name": self._get_string_value(
                    "General", "dataset_name", ["MNIST", "CIFAR-10"]
                ),
                "batch_size": self._get_int_value("General", "batch_size", min_val=1),
            }
        except Exception as e:
            raise ConfigurationError(f"Error in General configuration: {str(e)}")

    def get_training_config(self) -> Dict[str, Union[int, float]]:
        """
        Get training configuration parameters.

        Returns:
            Dictionary containing training parameters

        Raises:
            ConfigurationError: If required parameters are missing or invalid
        """
        try:
            return {
                "num_epochs": self._get_int_value("Training", "num_epochs", min_val=1),
                "learning_rate": self._get_float_value(
                    "Training", "learning_rate", min_val=0.0
                ),
            }
        except Exception as e:
            raise ConfigurationError(f"Error in Training configuration: {str(e)}")

    def get_image_patching_config(self) -> Dict[str, int]:
        """
        Get image patching configuration parameters.

        Returns:
            Dictionary containing image patching parameters

        Raises:
            ConfigurationError: If required parameters are missing or invalid
        """
        try:
            patch_size = self._get_int_value(
                "ImagePatching", "patch_size", min_val=1, max_val=28
            )
            img_size = self._get_int_value("ImagePatching", "img_size", min_val=1)
            in_channels = self._get_int_value(
                "ImagePatching", "in_channels", min_val=1, max_val=3
            )

            # Validate that img_size is divisible by patch_size
            if img_size % patch_size != 0:
                raise ConfigurationError(
                    f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
                )

            return {
                "patch_size": patch_size,
                "img_size": img_size,
                "in_channels": in_channels,
            }
        except Exception as e:
            raise ConfigurationError(f"Error in ImagePatching configuration: {str(e)}")

    def get_transformer_config(self) -> Dict[str, int]:
        """
        Get transformer encoder configuration parameters.

        Returns:
            Dictionary containing transformer parameters

        Raises:
            ConfigurationError: If required parameters are missing or invalid
        """
        try:
            embed_dim = self._get_int_value(
                "TransformerEncoder", "embed_dim", min_val=1
            )
            num_heads = self._get_int_value(
                "TransformerEncoder", "num_heads", min_val=1
            )
            num_layers = self._get_int_value(
                "TransformerEncoder", "num_layers", min_val=1
            )

            # Validate that embed_dim is divisible by num_heads
            if embed_dim % num_heads != 0:
                raise ConfigurationError(
                    f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                )

            return {
                "num_layers": num_layers,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
            }
        except Exception as e:
            raise ConfigurationError(
                f"Error in TransformerEncoder configuration: {str(e)}"
            )

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get complete model configuration for Vision Transformer.

        Args:
            model_name: Name of the model to configure

        Returns:
            Dictionary containing all model parameters

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if model_name != "VisionTransformer":
            raise ConfigurationError(
                f"Model '{model_name}' configuration not supported"
            )

        # Get all configuration sections
        image_config = self.get_image_patching_config()
        transformer_config = self.get_transformer_config()

        # Add number of classes based on dataset
        general_config = self.get_general_config()
        num_classes = (
            10 if general_config["dataset_name"] == "MNIST" else 10
        )  # Both MNIST and CIFAR-10 have 10 classes

        # Combine all configurations
        model_config = {
            **image_config,
            **transformer_config,
            "num_classes": num_classes,
        }

        return model_config

    def validate_config(self) -> bool:
        """
        Validate entire configuration.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If any validation fails
        """
        try:
            # Validate all sections
            general_config = self.get_general_config()
            self.get_training_config()
            self.get_image_patching_config()
            self.get_transformer_config()

            # Additional cross-section validations can be added here
            logging.info("Configuration validation successful")
            return True

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Unexpected error during validation: {str(e)}")


def load_and_validate_config(config_path: str = "config.ini") -> ConfigurationManager:
    """
    Convenience function to load and validate configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Validated ConfigurationManager instance

    Raises:
        ConfigurationError: If configuration is invalid
    """
    config_manager = ConfigurationManager(config_path)
    config_manager.validate_config()
    return config_manager
