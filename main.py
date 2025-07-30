import logging
import torch
import torch.nn as nn
import torch.optim as optim
from python.config_utils import ConfigurationManager, ConfigurationError
from python.VITs import ViTStandard
from python.dataset_utils import create_mnist_loaders, validate_mnist_compatibility

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransformerExperiment:
    def __init__(self, config_path='config.ini'):
        try:
            # Use ConfigurationManager for proper validation
            self.config_manager = ConfigurationManager(config_path)
            self.config_manager.validate_config()
            
            # Get general configuration
            general_config = self.config_manager.get_general_config()
            self.model_name = general_config['model_name']
            self.dataset_name = general_config['dataset_name']
            
            logging.info(f"Configuration loaded and validated from {config_path}")
            logging.info(f"Model: {self.model_name}, Dataset: {self.dataset_name}")
            
        except ConfigurationError as e:
            logging.error(f"Configuration error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading configuration: {str(e)}")
            raise

    def get_model_config(self):
        """Retrieves the configuration for the selected model."""
        try:
            return self.config_manager.get_model_config(self.model_name)
        except ConfigurationError as e:
            logging.error(f"Failed to get model configuration: {str(e)}")
            raise

    def select_model(self):
        """Selects and initializes the model based on the configuration."""
        try:
            model_config = self.get_model_config()
            logging.info(f"Selecting model: {self.model_name}")
            logging.info(f"Model configuration: {model_config}")

            if self.model_name == 'VisionTransformer':
                # Create ViTStandard model with configuration parameters
                model = ViTStandard(
                    img_size=model_config['img_size'],
                    patch_size=model_config['patch_size'],
                    in_channels=model_config['in_channels'],
                    num_classes=model_config['num_classes'],
                    embed_dim=model_config['embed_dim'],
                    num_layers=model_config['num_layers'],
                    num_heads=model_config['num_heads']
                )
                
                logging.info(f"ViTStandard model instantiated successfully")
                logging.info(f"Model parameters: {model.get_config()}")
                return model
                
            elif self.model_name == 'BasicTransformer':
                raise NotImplementedError("BasicTransformer model is not yet implemented.")
            else:
                raise NotImplementedError(f"Model '{self.model_name}' is not implemented.")
                
        except ConfigurationError as e:
            logging.error(f"Configuration error during model selection: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error instantiating model: {str(e)}")
            raise

    def load_dataset(self):
        """Loads the dataset based on the configuration."""
        try:
            logging.info(f"Loading dataset: {self.dataset_name}")
            
            if self.dataset_name == 'MNIST':
                # Get model configuration to ensure dataset compatibility
                model_config = self.get_model_config()
                
                # Create MNIST data loaders with model-compatible parameters
                train_loader, test_loader = create_mnist_loaders(
                    img_size=model_config['img_size'],
                    in_channels=model_config['in_channels'],
                    batch_size=32,  # Default batch size, could be made configurable
                    num_workers=0,  # Default for compatibility
                    download=True,
                    data_root='./data'
                )
                
                # Validate dataset compatibility with model configuration
                validate_mnist_compatibility(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    expected_img_size=model_config['img_size'],
                    expected_channels=model_config['in_channels']
                )
                
                logging.info(f"MNIST dataset loaded successfully")
                logging.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
                
                return train_loader, test_loader
                
            elif self.dataset_name == 'CIFAR-10':
                raise NotImplementedError("CIFAR-10 dataset loading is not yet implemented.")
            else:
                raise NotImplementedError(f"Dataset '{self.dataset_name}' is not implemented.")
                
        except ConfigurationError as e:
            logging.error(f"Configuration error during dataset loading: {str(e)}")
            raise
        except RuntimeError as e:
            logging.error(f"Dataset loading error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading dataset: {str(e)}")
            raise

    def run(self):
        """Runs the experiment."""
        try:
            logging.info("Starting experiment...")
            
            # Load dataset
            train_loader, test_loader = self.load_dataset()
            
            # Select model
            model = self.select_model()
            
            # Validate dataset compatibility with model
            logging.info("Validating dataset compatibility with model...")
            model.validate_dataset_compatibility(train_loader)
            model.validate_dataset_compatibility(test_loader)
            logging.info("Dataset compatibility validation successful")
            
            # Train and evaluate the model
            logging.info("Starting training and evaluation...")
            
            # Train the model
            trained_model = self.train_model(model, train_loader)
            
            # Evaluate the model with basic metrics
            accuracy = self.evaluate_model(trained_model, test_loader)
            
            # Generate comprehensive evaluation report
            self.generate_evaluation_report(trained_model, test_loader, save_report=True)
            
            logging.info(f"Training completed. Final test accuracy: {accuracy:.4f}")
            logging.info("Experiment finished successfully.")
            
        except (ConfigurationError, RuntimeError, ValueError) as e:
            logging.error(f"Experiment failed: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during experiment: {str(e)}")
            raise

    def train_model(self, model, train_loader, num_epochs=10, learning_rate=0.001):
        """
        Train the Vision Transformer model.
        
        Args:
            model: The ViT model to train
            train_loader: Training data loader
            num_epochs: Number of training epochs (default: 10)
            learning_rate: Learning rate for optimizer (default: 0.001)
            
        Returns:
            Trained model
        """
        try:
            # Set device (GPU if available, otherwise CPU)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            logging.info(f"Training on device: {device}")
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            model.train()
            total_batches = len(train_loader)
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                correct_predictions = 0
                total_samples = 0
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    # Move data to device
                    images, labels = images.to(device), labels.to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()
                    
                    # Log progress every 100 batches
                    if (batch_idx + 1) % 100 == 0:
                        batch_accuracy = 100.0 * correct_predictions / total_samples
                        logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
                                   f'Batch [{batch_idx+1}/{total_batches}], '
                                   f'Loss: {loss.item():.4f}, '
                                   f'Accuracy: {batch_accuracy:.2f}%')
                
                # Calculate epoch metrics
                avg_epoch_loss = epoch_loss / total_batches
                epoch_accuracy = 100.0 * correct_predictions / total_samples
                
                logging.info(f'Epoch [{epoch+1}/{num_epochs}] completed - '
                           f'Average Loss: {avg_epoch_loss:.4f}, '
                           f'Training Accuracy: {epoch_accuracy:.2f}%')
            
            logging.info("Training completed successfully")
            return model
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def evaluate_model(self, model, test_loader):
        """
        Evaluate the trained model on test data.
        
        Args:
            model: The trained model to evaluate
            test_loader: Test data loader
            
        Returns:
            Test accuracy as a float
        """
        try:
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Set model to evaluation mode
            model.eval()
            
            correct_predictions = 0
            total_samples = 0
            total_loss = 0.0
            criterion = nn.CrossEntropyLoss()
            
            # Disable gradient computation for evaluation
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    # Move data to device
                    images, labels = images.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Track metrics
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()
                    
                    # Log progress every 50 batches
                    if (batch_idx + 1) % 50 == 0:
                        current_accuracy = 100.0 * correct_predictions / total_samples
                        logging.info(f'Evaluation Batch [{batch_idx+1}/{len(test_loader)}], '
                                   f'Current Accuracy: {current_accuracy:.2f}%')
            
            # Calculate final metrics
            test_accuracy = correct_predictions / total_samples
            avg_test_loss = total_loss / len(test_loader)
            
            logging.info(f'Evaluation completed - '
                       f'Test Loss: {avg_test_loss:.4f}, '
                       f'Test Accuracy: {test_accuracy:.4f} ({100.0 * test_accuracy:.2f}%)')
            
            return test_accuracy
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

    def evaluate_model_comprehensive(self, model, test_loader):
        """
        Comprehensive model evaluation with detailed metrics reporting.
        
        Args:
            model: The trained model to evaluate
            test_loader: Test data loader
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        try:
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Set model to evaluation mode
            model.eval()
            
            # Initialize metrics tracking
            correct_predictions = 0
            total_samples = 0
            total_loss = 0.0
            class_correct = [0] * 10  # MNIST has 10 classes
            class_total = [0] * 10
            all_predictions = []
            all_labels = []
            
            criterion = nn.CrossEntropyLoss()
            
            # Disable gradient computation for evaluation
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    # Move data to device
                    images, labels = images.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Get predictions
                    _, predicted = torch.max(outputs, 1)
                    
                    # Track overall metrics
                    total_loss += loss.item()
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()
                    
                    # Track per-class metrics
                    for i in range(labels.size(0)):
                        label = labels[i].item()
                        class_total[label] += 1
                        if predicted[i] == labels[i]:
                            class_correct[label] += 1
                    
                    # Store predictions and labels for additional analysis
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Log progress
                    if (batch_idx + 1) % 50 == 0:
                        current_accuracy = 100.0 * correct_predictions / total_samples
                        logging.info(f'Comprehensive Evaluation Batch [{batch_idx+1}/{len(test_loader)}], '
                                   f'Current Accuracy: {current_accuracy:.2f}%')
            
            # Calculate overall metrics
            overall_accuracy = correct_predictions / total_samples
            avg_loss = total_loss / len(test_loader)
            
            # Calculate per-class accuracies
            class_accuracies = {}
            for i in range(10):
                if class_total[i] > 0:
                    class_accuracies[f'class_{i}'] = class_correct[i] / class_total[i]
                else:
                    class_accuracies[f'class_{i}'] = 0.0
            
            # Create comprehensive results dictionary
            results = {
                'overall_accuracy': overall_accuracy,
                'overall_accuracy_percent': 100.0 * overall_accuracy,
                'average_loss': avg_loss,
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'class_accuracies': class_accuracies,
                'class_totals': {f'class_{i}': class_total[i] for i in range(10)},
                'class_correct': {f'class_{i}': class_correct[i] for i in range(10)}
            }
            
            # Log comprehensive results
            logging.info("=== Comprehensive Evaluation Results ===")
            logging.info(f"Overall Accuracy: {results['overall_accuracy_percent']:.2f}% "
                       f"({correct_predictions}/{total_samples})")
            logging.info(f"Average Loss: {avg_loss:.4f}")
            
            logging.info("Per-class Accuracies:")
            for i in range(10):
                class_acc = class_accuracies[f'class_{i}']
                class_count = class_total[i]
                logging.info(f"  Class {i}: {100.0 * class_acc:.2f}% ({class_correct[i]}/{class_count})")
            
            return results
            
        except Exception as e:
            logging.error(f"Error during comprehensive evaluation: {str(e)}")
            raise

    def generate_evaluation_report(self, model, test_loader, save_report=False, report_path="evaluation_report.txt"):
        """
        Generate a detailed evaluation report.
        
        Args:
            model: The trained model to evaluate
            test_loader: Test data loader
            save_report: Whether to save report to file (default: False)
            report_path: Path to save the report (default: "evaluation_report.txt")
            
        Returns:
            String containing the evaluation report
        """
        try:
            # Get comprehensive evaluation results
            results = self.evaluate_model_comprehensive(model, test_loader)
            
            # Generate report
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("VISION TRANSFORMER EVALUATION REPORT")
            report_lines.append("=" * 60)
            report_lines.append("")
            
            # Model configuration
            model_config = model.get_config()
            report_lines.append("MODEL CONFIGURATION:")
            for key, value in model_config.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
            
            # Overall performance
            report_lines.append("OVERALL PERFORMANCE:")
            report_lines.append(f"  Test Accuracy: {results['overall_accuracy_percent']:.2f}%")
            report_lines.append(f"  Test Loss: {results['average_loss']:.4f}")
            report_lines.append(f"  Total Samples: {results['total_samples']}")
            report_lines.append(f"  Correct Predictions: {results['correct_predictions']}")
            report_lines.append("")
            
            # Per-class performance
            report_lines.append("PER-CLASS PERFORMANCE:")
            for i in range(10):
                class_acc = results['class_accuracies'][f'class_{i}']
                class_total = results['class_totals'][f'class_{i}']
                class_correct = results['class_correct'][f'class_{i}']
                report_lines.append(f"  Class {i}: {100.0 * class_acc:.2f}% ({class_correct}/{class_total})")
            report_lines.append("")
            
            # Performance analysis
            report_lines.append("PERFORMANCE ANALYSIS:")
            best_class = max(range(10), key=lambda i: results['class_accuracies'][f'class_{i}'])
            worst_class = min(range(10), key=lambda i: results['class_accuracies'][f'class_{i}'])
            
            report_lines.append(f"  Best performing class: {best_class} "
                              f"({100.0 * results['class_accuracies'][f'class_{best_class}']:.2f}%)")
            report_lines.append(f"  Worst performing class: {worst_class} "
                              f"({100.0 * results['class_accuracies'][f'class_{worst_class}']:.2f}%)")
            
            # Calculate standard deviation of class accuracies
            class_accs = [results['class_accuracies'][f'class_{i}'] for i in range(10)]
            mean_class_acc = sum(class_accs) / len(class_accs)
            std_class_acc = (sum((acc - mean_class_acc) ** 2 for acc in class_accs) / len(class_accs)) ** 0.5
            
            report_lines.append(f"  Mean class accuracy: {100.0 * mean_class_acc:.2f}%")
            report_lines.append(f"  Std dev of class accuracies: {100.0 * std_class_acc:.2f}%")
            report_lines.append("")
            
            report_lines.append("=" * 60)
            
            # Join all lines
            report = "\n".join(report_lines)
            
            # Save report if requested
            if save_report:
                with open(report_path, 'w') as f:
                    f.write(report)
                logging.info(f"Evaluation report saved to {report_path}")
            
            # Log the report
            logging.info("Generated evaluation report:")
            logging.info(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating evaluation report: {str(e)}")
            raise


if __name__ == "__main__":
    experiment = TransformerExperiment()
    experiment.run()
