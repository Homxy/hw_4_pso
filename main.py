import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def read_dataset(filepath):
    # Load and process the dataset
    data = pd.read_excel(filepath)
    features = data.iloc[:, [3, 6, 8, 10, 11, 12, 13, 14]].values
    target = data.iloc[:, 5].values  # Column 5 is the target (benzene concentration)
    return features, target

def standardize(data):
    # Standardize features
    features = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return features

def create_folds(features, target, fold_size=0.1):
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    split = int(len(features) * (1 - fold_size))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, X_val = features[train_idx], features[val_idx]
    y_train, y_val = target[train_idx], target[val_idx]
    return X_train, y_train, X_val, y_val

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.architecture = [input_dim] + hidden_dims + [output_dim]
        self.weights = [np.random.randn(self.architecture[i], self.architecture[i + 1]) * 0.1 
                        for i in range(len(self.architecture) - 1)]
        self.biases = [np.random.randn(1, self.architecture[i + 1]) * 0.1 
                       for i in range(len(self.architecture) - 1)]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.sigmoid(z))
        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        return z_out  # Regression output, no activation in the last layer

    def predict(self, X):
        return self.forward(X).flatten()  # Output as 1D array

class PSO(MLP):
    def __init__(self, pop_size, max_iterations, input_size, hidden_layers, output_size, 
                 inertia=0.7, cognitive=1.5, social=1.5):
        super().__init__(input_size, hidden_layers, output_size)
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.global_best_weights = None
        self.global_best_biases = None
        self.global_best_score = float('inf')

    def initialize_population(self):
        # Initialize population with random weights and velocities
        population = []
        velocities = []
        for _ in range(self.pop_size):
            model = MLP(self.architecture[0], self.architecture[1:-1], self.architecture[-1])
            velocity = [np.random.randn(*w.shape) * 0.1 for w in model.weights]
            population.append(model)
            velocities.append(velocity)
        return population, velocities

    def evaluate(self, model, X, y):
        predictions = model.predict(X)
        return np.mean(np.abs(predictions - y))

    def optimize(self, X_train, y_train, X_val, y_val):
        population, velocities = self.initialize_population()
        for model in population:
            score = self.evaluate(model, X_val, y_val)
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_weights = [w.copy() for w in model.weights]
                self.global_best_biases = [b.copy() for b in model.biases]

        for iteration in range(self.max_iterations):
            for i, model in enumerate(population):
                score = self.evaluate(model, X_val, y_val)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_weights = [w.copy() for w in model.weights]
                    self.global_best_biases = [b.copy() for b in model.biases]

                # Update velocity and position for each model
                for j in range(len(model.weights)):
                    velocities[i][j] = (
                        self.inertia * velocities[i][j] +
                        self.cognitive * np.random.rand() * (self.global_best_weights[j] - model.weights[j]) +
                        self.social * np.random.rand() * (self.global_best_weights[j] - model.weights[j])
                    )
                    model.weights[j] += velocities[i][j]

        # Set best model weights after training
        best_model = MLP(self.architecture[0], self.architecture[1:-1], self.architecture[-1])
        best_model.weights = self.global_best_weights
        best_model.biases = self.global_best_biases
        return best_model

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def cross_validate(features, target, fold_size=0.1):
    X_train, y_train, X_val, y_val = create_folds(features, target, fold_size)
    pso = PSO(
        pop_size=20, max_iterations=100, 
        input_size=X_train.shape[1], hidden_layers=[50], output_size=1
    )
    best_model = pso.optimize(X_train, y_train, X_val, y_val)
    y_pred = best_model.predict(X_val)
    fold_mae = calculate_mae(y_val, y_pred)
    
    print(f'Fold MAE: {fold_mae:.4f}')
    return fold_mae, y_val, y_pred

def main():
    filepath = 'AirQualityUCI.xlsx'
    features, target = read_dataset(filepath)
    features = standardize(features)

    mae_per_fold = []
    all_y_val, all_y_pred = [], []
    
    for _ in range(10):
        fold_mae, y_val, y_pred = cross_validate(features, target)
        mae_per_fold.append(fold_mae)
        all_y_val.extend(y_val)
        all_y_pred.extend(y_pred)

    avg_mae = np.mean(mae_per_fold)
    print(f'Final MAE: {avg_mae:.4f}')

    # Plotting MAE per fold
    plt.figure()
    plt.plot(mae_per_fold, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    plt.title('MAE per Fold')
    plt.show()

    # Plotting final Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(target, label='Actual', color='blue')
    plt.plot(all_y_pred, label='Predicted', color='red', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Benzene Concentration')
    plt.title('Final Actual vs Predicted - Validation Set')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
