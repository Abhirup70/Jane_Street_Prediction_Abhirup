# Jane Street Market Prediction

This project implements a machine learning pipeline for the Jane Street Market Prediction challenge, using PyTorch to build, train, and evaluate models for predicting market movements based on historical data.

## Dataset

The project uses a sample of the Jane Street Market Prediction dataset, containing 130 anonymized features. The original dataset was part of a Kaggle competition, but this project uses a sample available on GitHub.

Features include:
- 130 anonymized numerical features
- 'date' and 'ts_id' metadata
- 'weight' column indicating the importance of each sample

## Project Structure

- `data/` - Directory containing raw and processed data
- `models/` - Directory containing trained models and evaluation results
- `download_data.py` - Script to download sample data from GitHub
- `data_exploration.py` - Script to explore and analyze the dataset
- `data_preprocessing.py` - Script to preprocess the data (handling missing values, standardization, etc.)
- `model.py` - Implementation of machine learning models (MLP and Transformer)
- `train.py` - Script to train the models
- `evaluate.py` - Script to evaluate trained models
- `predict.py` - Script to make predictions on new data

## Models

The project implements two types of models:

1. **Multi-layer Perceptron (MLP)**: A feedforward neural network with multiple hidden layers, batch normalization, and dropout.
2. **Transformer**: A transformer-based model that leverages self-attention mechanisms to capture complex relationships between features.

## Setup and Installation

```bash
# Clone the repository
git clone <repository-url>
cd jane-street-market-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn torch tqdm

# Download sample data
python download_data.py

# Explore the dataset
python data_exploration.py

# Preprocess the data
python data_preprocessing.py
```

## Usage

### Training

To train the MLP model:

```bash
python train.py --model_type mlp --epochs 20 --batch_size 256
```

To train the Transformer model:

```bash
python train.py --model_type transformer --epochs 20 --batch_size 256
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model_type mlp
```

### Prediction

To make predictions on new data:

```bash
python predict.py --model_type mlp --data_path <path-to-data>
```

## Results

The MLP model achieved the following results on the validation set:
- Accuracy: 98.42%
- AUC: 0.9992
- Precision: 0.9824
- Recall: 0.9774
- F1 Score: 0.9799

Note: Since we're using synthetic targets for demonstration purposes (as the original targets are not available), these metrics are for illustrative purposes only.

## Future Improvements

- Implement more sophisticated feature engineering
- Explore other model architectures such as LSTMs or GRUs
- Add hyperparameter tuning to optimize model performance
- Implement ensemble methods to combine multiple models
- Add support for more realistic trading simulation and evaluation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction) competition on Kaggle
- Sample data from [flame0409/Jane-Street-Market-Prediction](https://github.com/flame0409/Jane-Street-Market-Prediction) 