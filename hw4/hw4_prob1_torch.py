import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Generate the Dataset
np.random.seed(42)

# Parameters
n_train = 1000  # Number of training samples
n_test = 10000  # Number of test samples
r_neg = 2       # Radius for class -1
r_pos = 4       # Radius for class +1
sigma = 1       # Standard deviation of noise

def generate_data(n, radius):
    theta = np.random.uniform(-np.pi, np.pi, n)  # Uniform distribution for angle
    noise = np.random.normal(0, sigma, (n, 2))   # Gaussian noise
    x = radius * np.column_stack((np.cos(theta), np.sin(theta))) + noise
    return x

# Generate training data
X_train_neg = generate_data(n_train // 2, r_neg)
X_train_pos = generate_data(n_train // 2, r_pos)
X_train = np.vstack((X_train_neg, X_train_pos))
y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))  # Convert labels to 0 and 1

# Generate test data
X_test_neg = generate_data(n_test // 2, r_neg)
X_test_pos = generate_data(n_test // 2, r_pos)
X_test = np.vstack((X_test_neg, X_test_pos))
y_test = np.array([0] * (n_test // 2) + [1] * (n_test // 2))  # Convert labels to 0 and 1

# Visualize the training data
plt.figure(figsize=(8, 8))
plt.scatter(X_train_neg[:, 0], X_train_neg[:, 1], color='red', label='Class -1')
plt.scatter(X_train_pos[:, 0], X_train_pos[:, 1], color='blue', label='Class 1')
plt.title("Training Data")
plt.legend()
plt.show()

# Visualize the test data
plt.figure(figsize=(8, 8))
plt.scatter(X_test_neg[:, 0], X_test_neg[:, 1], color='red', label='Class -1')
plt.scatter(X_test_pos[:, 0], X_test_pos[:, 1], color='blue', label='Class 1')
plt.title("Test Data")
plt.legend()
plt.show()

# --------------------------------

# Step 2: Train SVM with scikit-learn
# Preprocess data using scikit-learn
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM with GridSearchCV
svm = SVC(kernel='rbf')
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=10, scoring='accuracy', return_train_score=True)
grid_search_svm.fit(X_train_scaled, y_train)

# Print best parameters and test accuracy
best_svm = grid_search_svm.best_estimator_
print(f"Best SVM Parameters: {grid_search_svm.best_params_}")  # This line prints the best hyperparameters
y_pred_svm = best_svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Test Accuracy: {accuracy_svm * 100:.2f}%")


# Extract GridSearch results for visualization
results = grid_search_svm.cv_results_
mean_test_scores = results['mean_test_score'].reshape(len(param_grid_svm['C']), len(param_grid_svm['gamma']))

# Plot SVM Hyperparameter Heatmap
plt.figure(figsize=(10, 6))
plt.imshow(mean_test_scores, interpolation='nearest', cmap='viridis')
plt.title("SVM Hyperparameter Tuning Heatmap")
plt.xlabel("Gamma")
plt.ylabel("C")
plt.xticks(np.arange(len(param_grid_svm['gamma'])), param_grid_svm['gamma'])
plt.yticks(np.arange(len(param_grid_svm['C'])), param_grid_svm['C'])
plt.colorbar(label='Validation Accuracy')
plt.show()

# --------------------------------

# Step 3: Train MLP with PyTorch
# Define Stabilized Quadratic Activation
class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x ** 2 / (1 + torch.abs(x))

# Define MLP Model
class MLPWithQuadratic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPWithQuadratic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.quadratic = QuadraticActivation()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.quadratic(x)
        x = self.fc2(x)
        return x

# Parameters
input_size = 2
# hidden_size = 50  # Increased hidden size
output_size = 1

# Perform 10-fold cross-validation for hidden size tuning
hidden_sizes = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210]  # Possible hidden layer sizes
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

best_hidden_size = None
best_val_accuracy = 0
hidden_size_accuracies = []  # Store average validation accuracies for visualization

# Cross-validation loop
for hidden_size in hidden_sizes:
    fold_accuracies = []
    for train_idx, val_idx in kfold.split(X_train_scaled):
        # Prepare training and validation data for this fold
        X_train_fold = torch.tensor(X_train_scaled[train_idx], dtype=torch.float32)
        y_train_fold = torch.tensor(y_train[train_idx], dtype=torch.float32).view(-1, 1)
        X_val_fold = torch.tensor(X_train_scaled[val_idx], dtype=torch.float32)
        y_val_fold = torch.tensor(y_train[val_idx], dtype=torch.float32).view(-1, 1)

        # Initialize model for the current hidden size
        model = MLPWithQuadratic(input_size=2, hidden_size=hidden_size, output_size=1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Train for fewer epochs to evaluate the hidden size
        for epoch in range(200):  # Cross-validation training loop
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_fold)
            loss = criterion(logits, y_train_fold)
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        with torch.no_grad():
            logits = model(X_val_fold)
            val_preds = (torch.sigmoid(logits) > 0.5).numpy().flatten()
            val_accuracy = accuracy_score(y_train[val_idx], val_preds)
            fold_accuracies.append(val_accuracy)

    # Average accuracy for this hidden size
    avg_val_accuracy = np.mean(fold_accuracies)
    hidden_size_accuracies.append(avg_val_accuracy)  # Store for visualization
    print(f"Hidden Size: {hidden_size}, Average Validation Accuracy: {avg_val_accuracy * 100:.2f}%")

    # Update the best hidden size if this one performs better
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        best_hidden_size = hidden_size

print(f"Best Hidden Size: {best_hidden_size}, Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")

# Visualize MLP Hidden Size Selection
plt.figure(figsize=(10, 6))
plt.plot(hidden_sizes, hidden_size_accuracies, marker='o', linestyle='-', color='b', label='Validation Accuracy')
plt.axvline(x=best_hidden_size, color='r', linestyle='--', label=f'Best Hidden Size = {best_hidden_size}')
plt.title("MLP Hyperparameter Tuning: Hidden Size vs. Validation Accuracy")
plt.xlabel("Hidden Size")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Use the best hidden size for final training
model = MLPWithQuadratic(input_size=2, hidden_size=best_hidden_size, output_size=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Convert data to PyTorch tensors (unchanged)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# --------------------------------

# Final Training Loop (2000 epochs)
n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    logits = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    with torch.no_grad():
        train_preds = (torch.sigmoid(logits) > 0.5).numpy().flatten()
        train_accuracy = accuracy_score(y_train, train_preds)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, Train Accuracy: {train_accuracy * 100:.2f}%")

# --------------------------------

# Evaluate MLP on test data (unchanged)
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    y_pred_mlp = (torch.sigmoid(logits) > 0.5).numpy().flatten()
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    print(f"MLP Test Accuracy: {accuracy_mlp * 100:.2f}%")

# --------------------------------

def plot_decision_boundary(model, X, y, title, scaler=None, is_pytorch=False, step_size=0.05):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Scale grid if necessary
    if scaler:
        grid = scaler.transform(grid)
    
    # Predict on the grid
    if is_pytorch:
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        with torch.no_grad():
            predictions = (torch.sigmoid(model(grid_tensor)) > 0.5).numpy().flatten()
    else:
        predictions = model.predict(grid)
    
    # Reshape predictions to match the mesh grid
    predictions = predictions.reshape(xx.shape)
    
    # Plot decision regions
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Add contour line for decision boundary
    plt.contour(xx, yy, predictions, levels=[0.5], colors='black', linewidths=2)
    
    # Plot data points with reduced opacity
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm, alpha=0.5)
    plt.title(title)
    plt.show()


plot_decision_boundary(best_svm, X_test, y_test, title="SVM Decision Boundary", scaler=scaler)
plot_decision_boundary(model, X_test, y_test, title="MLP Decision Boundary", scaler=scaler, is_pytorch=True)

