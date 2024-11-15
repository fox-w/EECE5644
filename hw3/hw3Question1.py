import numpy as np
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Data Distribution Specification
# -----------------------------

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

NUM_CLASSES = 4       # Number of classes
FEATURE_DIM = 3       # Dimensionality of feature space

# Adjusted for right separation
class_means = np.array([
    [0, 0, 0],        # Class 0
    [3, 3, 3],        # Class 1
    [3, 0, 3],        # Class 2
    [0, 3, 3]         # Class 3
])

# Adjusted for right overlap
covariance_value = 1.5 

class_covariances = np.array([
    [[covariance_value, 0, 0],
     [0, covariance_value, 0],
     [0, 0, covariance_value]],
    
    [[covariance_value, 0, 0],
     [0, covariance_value, 0],
     [0, 0, covariance_value]],
    
    [[covariance_value, 0, 0],
     [0, covariance_value, 0],
     [0, 0, covariance_value]],
    
    [[covariance_value, 0, 0],
     [0, covariance_value, 0],
     [0, 0, covariance_value]]
])

# -----------------------------
# 2. Data Generation
# -----------------------------

def generate_dataset(num_samples, means, covariances, num_classes):
    """
    Generates dataset with specified number of samples, class means, and covariances.
    """
    samples_per_class = num_samples // num_classes
    remainder = num_samples % num_classes
    features = []
    labels = []
    
    for class_label in range(num_classes):
        # Distribute the remainder samples among the first 'remainder' classes
        num_samples_class = samples_per_class + (1 if class_label < remainder else 0)
        
        class_features = np.random.multivariate_normal(
            mean=means[class_label],
            cov=covariances[class_label],
            size=num_samples_class
        )
        features.append(class_features)
        labels += [class_label] * num_samples_class
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    # Shuffle the dataset
    shuffled_indices = np.arange(num_samples)
    np.random.shuffle(shuffled_indices)
    
    return features[shuffled_indices], labels[shuffled_indices]

# Generate a large test dataset for performance evaluation
print("Generating test dataset...")
test_features, test_labels = generate_dataset(100000, class_means, class_covariances, NUM_CLASSES)

# -----------------------------
# 3. Theoretically Optimal Classifier
# -----------------------------

def evaluate_theoretical_classifier(X, y_true, means, covariances, num_classes):
    """
    Evaluate the theoretical optimal classifier based on true data distributions
    and return empirical probability of error.
    """
    posteriors = np.zeros((X.shape[0], num_classes))
    for class_label in range(num_classes):
        rv = multivariate_normal(mean=means[class_label], cov=covariances[class_label])
        posteriors[:, class_label] = rv.pdf(X)
    predicted_labels = np.argmax(posteriors, axis=1)
    error_rate = np.mean(predicted_labels != y_true)
    return error_rate

print("Evaluating Theoretical Optimal Classifier...")
theoretical_error_rate = evaluate_theoretical_classifier(
    test_features, test_labels, class_means, class_covariances, NUM_CLASSES
)
print(f"Theoretical Optimal Classifier Error Rate: {theoretical_error_rate * 100:.2f}%")

# -----------------------------
# 4. MLP Structure with Smooth-Ramp Activation
# -----------------------------

class MultilayerPerceptron(nn.Module):
    """
    2-layer Multilayer Perceptron (MLP) with 1 hidden layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation_function='elu'):
        super(MultilayerPerceptron, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        
        # Choose activation function
        if activation_function.lower() == 'elu':
            self.activation = nn.ELU()
        elif activation_function.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation_function.lower() == 'smooth_relu':
            self.activation = nn.Softplus()
        else:
            raise ValueError("Unsupported activation function. Choose 'elu', 'relu', or 'smooth_relu'.")
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

# -----------------------------
# 5. Model Order Selection via Cross-Validation
# -----------------------------

def select_optimal_hidden_units(features, labels, hidden_layer_sizes, num_folds=10, activation='elu'):
    """
    Perform cross-validation to select the optimal number of hidden units.
    """
    optimal_hidden_size = hidden_layer_sizes[0]
    highest_cv_accuracy = 0.0
    
    # Determine the minimum number of samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples_per_class = counts.min()
    
    # Adjust number of splits based on minimum samples per class
    adjusted_n_splits = min(num_folds, min_samples_per_class)
    if adjusted_n_splits < num_folds:
        print(f"Adjusted number of cross-validation folds to {adjusted_n_splits} based on class distribution.")
    
    if adjusted_n_splits < 2:
        print("Not enough samples per class for cross-validation. Using Leave-One-Out CV.")
        stratified_kfold = LeaveOneOut()
    else:
        stratified_kfold = StratifiedKFold(n_splits=adjusted_n_splits, shuffle=True, random_state=42)
    
    for hidden_size in hidden_layer_sizes:
        cv_accuracies = []
        if isinstance(stratified_kfold, StratifiedKFold):
            splits = stratified_kfold.split(features, labels)
        else:
            splits = stratified_kfold.split(features)
        
        for train_indices, val_indices in splits:
            X_train_cv, X_val_cv = features[train_indices], features[val_indices]
            y_train_cv, y_val_cv = labels[train_indices], labels[val_indices]
            
            # Standardize data based on training fold
            scaler_cv = StandardScaler()
            X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
            X_val_cv_scaled = scaler_cv.transform(X_val_cv)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train_cv_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_cv, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val_cv_scaled, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_cv, dtype=torch.long)
            
            # Create DataLoader for training
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            # Initialize the MLP model
            mlp_model = MultilayerPerceptron(
                input_dim=FEATURE_DIM,
                hidden_dim=hidden_size,
                output_dim=NUM_CLASSES,
                activation_function=activation
            )
            
            # Define loss function and optimizer
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
            
            # Training loop
            mlp_model.train()
            for epoch in range(50):  # Number of epochs can be adjusted
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = mlp_model(batch_X)
                    loss = loss_function(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation on validation set
            mlp_model.eval()
            with torch.no_grad():
                outputs = mlp_model(X_val_tensor)
                _, preds = torch.max(outputs, 1)
                accuracy = accuracy_score(y_val_tensor.numpy(), preds.numpy())
                cv_accuracies.append(accuracy)
        
        mean_cv_accuracy = np.mean(cv_accuracies)
        print(f"Hidden Units: {hidden_size}, Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")
        
        # Update optimal hidden size if current is better
        if mean_cv_accuracy > highest_cv_accuracy:
            highest_cv_accuracy = mean_cv_accuracy
            optimal_hidden_size = hidden_size
    
    return optimal_hidden_size

# -----------------------------
# 6. Model Training with Multiple Initializations
# -----------------------------

def train_final_mlp_model(features, labels, hidden_size, activation='elu', num_trials=5, num_epochs=100):
    """
    Train the final MLP model with selected number of hidden units.
    Perform multiple trials to mitigate local optimums.
    """
    best_model = None
    best_training_accuracy = 0.0
    best_scaler = None
    
    for trial in range(num_trials):
        # Set different random seeds for each trial
        torch.manual_seed(42 + trial)
        
        # Initialize the MLP model
        mlp_model = MultilayerPerceptron(
            input_dim=FEATURE_DIM,
            hidden_dim=hidden_size,
            output_dim=NUM_CLASSES,
            activation_function=activation
        )
        
        # Define loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
        
        # Standardize the entire training data
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Create DataLoader for training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Training loop
        mlp_model.train()
        for epoch in range(num_epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = mlp_model(batch_X)
                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Calculate training accuracy
        mlp_model.eval()
        with torch.no_grad():
            outputs = mlp_model(X_train_tensor)
            _, predicted = torch.max(outputs, 1)
            training_accuracy = accuracy_score(y_train_tensor.numpy(), predicted.numpy())
        
        # Select the model with the highest training accuracy
        if training_accuracy > best_training_accuracy:
            best_training_accuracy = training_accuracy
            best_model = mlp_model
            best_scaler = scaler
    
    return best_model, best_training_accuracy, best_scaler

# -----------------------------
# 7. Performance Assessment
# -----------------------------

def assess_model_performance(model, scaler, test_features, test_labels):
    """
    Assess the performance of the trained MLP model on the test dataset.
    """
    model.eval()
    with torch.no_grad():
        # Scale the test features using the training scaler
        test_features_scaled = scaler.transform(test_features)
        X_test_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
        
        # Forward pass
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        
        # Calculate error rate
        error_rate = np.mean(predicted.numpy() != test_labels)
    
    return error_rate

# -----------------------------
# 8. Results Visualization and Reporting
# -----------------------------

def plot_error_rates(training_sizes, mlp_error_rates, theoretical_error):
    """
    Plot the empirical probability of error for MLP classifiers versus training sample sizes.
    """
    plt.figure(figsize=(10, 6))
    plt.semilogx(training_sizes, [rate * 100 for rate in mlp_error_rates], marker='o', label='MLP Classifier Error Rate')
    plt.axhline(y=theoretical_error * 100, color='r', linestyle='--', label='Theoretical Optimal Error Rate')
    plt.xlabel('Number of Training Samples (log scale)')
    plt.ylabel('Test Set Error Rate (%)')
    plt.title('Test Set Error Rate vs. Number of Training Samples')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def plot_hidden_units(training_sizes, hidden_units_selected):
    """
    Plot the number of hidden units selected versus training sample sizes.
    """
    sizes_sorted = sorted(hidden_units_selected.keys())
    hidden_units = [hidden_units_selected[size] for size in sizes_sorted]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes_sorted, hidden_units, marker='s', color='g')
    plt.xscale('log')
    plt.xlabel('Number of Training Samples (log scale)')
    plt.ylabel('Selected Number of Hidden Units')
    plt.title('Selected Number of Hidden Units vs. Training Set Size')
    plt.grid(True, which="both", ls="--")
    plt.show()

# -----------------------------
# 9. Execution
# -----------------------------

# Define training dataset sizes
training_dataset_sizes = [10, 100, 500, 1000, 5000, 10000]

# Define range of hidden layer sizes to evaluate (adjusted to [10, 20, 30, 40, 50])
hidden_layer_sizes = [10, 20, 30, 40, 50]

# Containers to store results
mlp_models = {}
hidden_units_selected = {}
mlp_error_rates = []
scalers_used = {}

for size in training_dataset_sizes:
    print(f"\nTraining with {size} samples...")
    
    # Generate training dataset
    train_features, train_labels = generate_dataset(size, class_means, class_covariances, NUM_CLASSES)
    train_features, train_labels = shuffle(train_features, train_labels, random_state=42)
    
    # Model Order Selection via Cross-Validation
    print("Selecting best number of hidden units via cross-validation...")
    optimal_hidden_units = select_optimal_hidden_units(
        train_features,
        train_labels,
        hidden_layer_sizes,
        num_folds=10,
        activation='elu'
    )
    hidden_units_selected[size] = optimal_hidden_units
    print(f"Selected Hidden Units: {optimal_hidden_units}")
    
    # Train final MLP model with the selected number of hidden units
    print("Training final MLP model with selected hidden units...")
    final_model, training_accuracy, scaler = train_final_mlp_model(
        train_features,
        train_labels,
        hidden_size=optimal_hidden_units,
        activation='elu',
        num_trials=5,
        num_epochs=100
    )
    mlp_models[size] = final_model
    scalers_used[size] = scaler
    print(f"Best Training Accuracy: {training_accuracy * 100:.2f}%")
    
    # Evaluate the trained MLP model on the test set
    print("Evaluating model on test set...")
    test_error_rate = assess_model_performance(
        final_model,
        scaler,
        test_features,
        test_labels
    )
    mlp_error_rates.append(test_error_rate)
    print(f"Test Set Error Rate: {test_error_rate * 100:.2f}%")

# Re-evaluate Theoretical Optimal Classifier (to confirm consistency)
print("\nRe-evaluating Theoretical Optimal Classifier with Adjusted Data Distribution...")
theoretical_error_rate = evaluate_theoretical_classifier(
    test_features,
    test_labels,
    class_means,
    class_covariances,
    NUM_CLASSES
)
print(f"Theoretical Optimal Classifier Error Rate: {theoretical_error_rate * 100:.2f}%")

# -----------------------------
# 10. Plot Results
# -----------------------------

plot_error_rates(training_dataset_sizes, mlp_error_rates, theoretical_error_rate)
plot_hidden_units(training_dataset_sizes, hidden_units_selected)
