# CoreTempAI Package Documentation

This package provides the core functionality for temperature prediction using physics-based neural network and operator architectures.

## Package Components

### Data Processing

#### Data Generation (`data_generation/`)
- CFD simulation automation using Ansys Fluent
- Parameter space sampling
- Simulation monitoring and validation

#### Data Preprocessing (`data_preprocessing/`)
- Data cleaning and normalization
- Time series processing
- Dataset creation and validation

#### Data Aggregation (`data_aggregation/`)
- Multi-source data integration
- Data quality checks
- Dataset versioning

### Models (`model/`)

#### Neural Network Models
- Feedforward neural networks for scalar inputs
- Custom loss functions incorporating physical constraints
- Model architecture optimization

#### Neural Operator
- Processes both scalar and temporal inputs
- Self-attention mechanisms for temporal dependencies
- Physics-informed regularization

### Training (`training/`)

#### Training Features
- Multi-GPU support
- Automatic mixed precision
- Gradient clipping and normalization
- Early stopping and model checkpointing

#### Experiment Tracking
- Integration with Weights & Biases
- Comprehensive metric logging
- Hyperparameter optimization

### Input Parameters (`input_parameters/`)
- Parameter validation and preprocessing
- Input data normalization
- Feature selection utilities

### Utilities (`utils/`)
- Visualization tools
- Metric calculation
- Model evaluation
- Data validation

## Usage Examples

### Basic Usage

```python
from CoreTempAI.model import DiFormer
from CoreTempAI.config import DiFormerConfig
from CoreTempAI.data_preprocessing import DataProcessor

# Initialize data processor
processor = DataProcessor()

# Load and preprocess data
data = processor.load_data("path/to/data")
processed_data = processor.preprocess(data)

# Initialize and train model
config = DiFormerConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_layers=4
)
model = DiFormer(config)
model.train(processed_data)

# Make predictions
predictions = model.predict(test_data)
```

### Advanced Configuration

```python
from CoreTempAI.config import DiFormerConfig

config = DiFormerConfig(
    # Model architecture
    hidden_dim=256,
    num_heads=8,
    dropout_rate=0.1,
    
    # Training parameters
    learning_rate=1e-4,
    weight_decay=1e-6,
    gradient_clip_val=1.0,
    
    # Physics-informed parameters
    physics_loss_weight=0.1,
    conservation_loss_weight=0.05
)
```

### Custom Training Loop

```python
from CoreTempAI.training import Trainer
from CoreTempAI.utils import setup_wandb

# Initialize wandb logging
setup_wandb(project="core-temp-prediction")

# Create custom trainer
trainer = Trainer(
    model=model,
    config=config,
    use_gpu=True,
    distributed=True
)

# Train with custom callbacks
trainer.train(
    train_data=train_loader,
    val_data=val_loader,
    epochs=100,
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint(save_best=True)
    ]
)
```

## Configuration Options

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hidden_dim | int | 256 | Hidden layer dimension |
| num_layers | int | 4 | Number of transformer layers |
| num_heads | int | 8 | Number of attention heads |
| dropout_rate | float | 0.1 | Dropout rate |
| activation | str | 'gelu' | Activation function |

### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| learning_rate | float | 1e-4 | Initial learning rate |
| batch_size | int | 32 | Training batch size |
| max_epochs | int | 100 | Maximum training epochs |
| early_stopping_patience | int | 10 | Patience for early stopping |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## License

MIT License - see the LICENSE file for details. 