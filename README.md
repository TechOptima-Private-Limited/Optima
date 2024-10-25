# Optima - A Versatile Library for ML and NLP Model Training

Optima is a Python library designed to simplify and optimize traditional Machine Learning (ML) and Natural Language Processing (NLP) workflows. With an easy-to-use interface, Optima allows you to prepare datasets, train models, and evaluate performance for both ML and large language models (LLMs). This library supports scikit-learn models as well as transformer-based models from Hugging Face.

## Features

- **Unified Interface**: Train and evaluate both traditional ML models and transformer-based NLP models.
- **Data Preparation**: Quickly load, split, and prepare data for training.
- **Tokenizer Integration**: Easily tokenize text datasets using Hugging Face's transformers for NLP tasks.
- **Model Training**: Train both ML models (e.g., scikit-learn) and large language models using Trainer from Hugging Face.
- **Scalable Evaluations**: Evaluate trained models and get performance metrics like accuracy.

## Installation

You can install Optima using pip:

```sh
pip install Optima
```

## Usage

### 1. Import and Initialize Optima

Optima can be used for both ML models (e.g., scikit-learn classifiers) and NLP models (e.g., transformers). Here is how you can get started:

```python
from optima import Optima
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForSequenceClassification

# Example ML Model
ml_model = RandomForestClassifier()
optima_ml = Optima(model=ml_model, model_type='ml')

# Example NLP Model
llm_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
optima_nlp = Optima(model=llm_model, model_type='llm', model_name='distilbert-base-uncased')
```

### 2. Prepare Data

For ML models, Optima can load and split datasets like `digits` from scikit-learn:

```python
# Prepare data for ML model
X_train, X_test, y_train, y_test = optima_ml.prepare_data(dataset='digits')
```

For NLP models, you can load datasets from Hugging Face's `datasets` library:

```python
# Prepare data for NLP model
nlp_dataset = optima_nlp.prepare_data(dataset='imdb')  # e.g., IMDB movie reviews dataset
```

### 3. Tokenize Data (For NLP Models)

If you're working with NLP models, you need to tokenize the data before training:

```python
# Tokenize NLP dataset
tokenized_dataset = optima_nlp.tokenize_data(nlp_dataset)
```

### 4. Train the Model

You can train both ML and NLP models using the `train_model()` method:

```python
# Train ML model
optima_ml.train_model(X_train, y_train)

# Train NLP model
optima_nlp.train_model(X_train=tokenized_dataset['train'])
```

### 5. Evaluate the Model

Evaluate the performance of your trained model:

```python
# Evaluate ML model
accuracy = optima_ml.evaluate_model(X_test, y_test)
print(f"ML Model Accuracy: {accuracy:.2f}")

# Evaluate NLP model
results = optima_nlp.evaluate_model(X_test=tokenized_dataset['test'])
print("NLP Model Evaluation:", results)
```

## Requirements

Optima depends on several popular Python packages for ML and NLP tasks:

- `scikit-learn`
- `transformers`
- `datasets`
- `torch`

To install these requirements, you can use the following command:

```sh
pip install scikit-learn transformers datasets torch
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Vishwanath Akuthota
- Ganesh Thota
- Krishna Avula

## Contributing

We welcome contributions to improve Optima. Please feel free to submit issues and pull requests on the [GitHub repository](https://github.com/TechOptima-Private-Limited/Optima).