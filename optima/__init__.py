from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

class Optima:
    def __init__(self, model, model_name='model', model_type='ml'):
        self.model = model
        self.model_name = model_name
        self.model_type = model_type
        if self.model_type == 'llm':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self, dataset='digits', test_size=0.2, random_state=42):
        if self.model_type == 'ml':
            if dataset == 'digits':
                data = load_digits()
            else:
                raise NotImplementedError(f"Dataset '{dataset}' is not supported")
            X_train, X_test, y_train, y_test = train_test_split(
                data.data, data.target, test_size=test_size, random_state=random_state
            )
            return X_train, X_test, y_train, y_test
        elif self.model_type == 'llm':
            dataset = load_dataset(dataset)
            return dataset

    def tokenize_data(self, dataset):
        if self.model_type == 'llm':
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", truncation=True)

            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            tokenized_datasets = tokenized_datasets.remove_columns(["text"])
            tokenized_datasets.set_format("torch")
            return tokenized_datasets

    def train_model(self, X_train, y_train=None):
        if self.model_type == 'ml':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            self.model.fit(X_train, y_train)
            self.scaler = scaler  # Save the scaler for later use
        elif self.model_type == 'llm':
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=3,
                weight_decay=0.01,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=X_train,
                eval_dataset=y_train,
            )

            trainer.train()
            self.trainer = trainer  # Save the trainer for evaluation

    def evaluate_model(self, X_test, y_test=None):
        if self.model_type == 'ml':
            if not hasattr(self, 'model'):
                raise RuntimeError("You need to train the model before evaluating it.")
            X_test = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        elif self.model_type == 'llm':
            results = self.trainer.evaluate()
            return results



