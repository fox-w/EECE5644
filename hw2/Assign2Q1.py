import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Parameters for the Gaussian components
m01 = np.array([-0.9, -1.1])
m02 = np.array([0.8, 0.75])
m11 = np.array([-1.1, 0.9])
m12 = np.array([0.9, -0.75])
cov = np.array([[0.75, 0], [0, 1.25]])  # Covariance is the same for all components

# Class priors
prior_L0 = 0.6
prior_L1 = 0.4

def generate_dataset(n_samples):
    """Generate dataset with given number of samples."""
    n_L0 = int(n_samples * prior_L0)
    n_L1 = n_samples - n_L0
    
    # Generate samples for each class and each subgroup
    X_L0_1 = multivariate_normal.rvs(mean=m01, cov=cov, size=n_L0 // 2)
    X_L0_2 = multivariate_normal.rvs(mean=m02, cov=cov, size=n_L0 // 2)
    X_L1_1 = multivariate_normal.rvs(mean=m11, cov=cov, size=n_L1 // 2)
    X_L1_2 = multivariate_normal.rvs(mean=m12, cov=cov, size=n_L1 // 2)
    
    # Combine samples and labels
    X_L0 = np.vstack((X_L0_1, X_L0_2))
    X_L1 = np.vstack((X_L1_1, X_L1_2))
    y_L0 = np.zeros(n_L0)
    y_L1 = np.ones(n_L1)
    
    X = np.vstack((X_L0, X_L1))
    y = np.hstack((y_L0, y_L1))
    
    # Shuffle dataset to simulate real data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]

# Generate the datasets
D20_train, y20_train = generate_dataset(20)
D200_train, y200_train = generate_dataset(200)
D2000_train, y2000_train = generate_dataset(2000)
D10K_validate, y10K_validate = generate_dataset(10000)

# Class-conditional probabilities for optimal classifier
def p_x_given_L0(x):
    return 0.5 * multivariate_normal.pdf(x, mean=m01, cov=cov) + 0.5 * multivariate_normal.pdf(x, mean=m02, cov=cov)

def p_x_given_L1(x):
    return 0.5 * multivariate_normal.pdf(x, mean=m11, cov=cov) + 0.5 * multivariate_normal.pdf(x, mean=m12, cov=cov)

# Posterior probabilities using Bayes' rule - form of 
def posterior_L0(x):
    return (prior_L0 * p_x_given_L0(x)) / (prior_L0 * p_x_given_L0(x) + prior_L1 * p_x_given_L1(x))

def posterior_L1(x):
    return (prior_L1 * p_x_given_L1(x)) / (prior_L0 * p_x_given_L0(x) + prior_L1 * p_x_given_L1(x))

# Theoretically optimal classifier
def classify_optimal(x):
    return 0 if posterior_L0(x) > posterior_L1(x) else 1

# Applying optimal classifier to validation set
predicted_optimal = np.array([classify_optimal(x) for x in D10K_validate])
min_p_error = np.mean(predicted_optimal != y10K_validate)
print(f"Minimum Probability of Error (min-P(error)): {min_p_error:.4f}")

# Calculate posterier scores to be used for ROC curve
posterior_scores = np.array([posterior_L1(x) for x in D10K_validate])

# ROC curve and AUC for optimal classifier
fpr, tpr, thresholds = roc_curve(y10K_validate, posterior_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve for theoretically optimal classifier
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.scatter([fpr[np.argmax(tpr - fpr)]], [tpr[np.argmax(tpr - fpr)]], color='red', label='Min-P(error)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Theoretically Optimal Classifier')
plt.legend()
plt.grid()
plt.show()

# Decision boundary visualization for the theoretically optimal classifier
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z_optimal = np.array([classify_optimal(x) for x in grid_points]).reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z_optimal, alpha=0.3, cmap='coolwarm')
# Plot the decision boundary using contour lines
plt.contourf(xx, yy, Z_optimal, alpha=0.2, cmap='coolwarm', levels=1)  # Color shading for the decision regions
plt.contour(xx, yy, Z_optimal, colors='black', linewidths=0.5)          # Black contour lines for the boundary
plt.scatter(D10K_validate[:, 0], D10K_validate[:, 1], c=y10K_validate, edgecolor='k', cmap='coolwarm', s=10)
plt.title('Decision Boundary for Theoretically Optimal Classifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()




# PART 2

# Logistic Regression for approximations
def train_logistic_model(X_train, y_train, is_quadratic=False):
    if is_quadratic:
        # Quadratic feature transformation
        X_train = np.hstack((X_train, X_train[:, 0:1] ** 2, X_train[:, 0:1] * X_train[:, 1:2], X_train[:, 1:2] ** 2))
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Train logistic-linear and logistic-quadratic models with different training sets
linear_models = [train_logistic_model(D20_train, y20_train),
                 train_logistic_model(D200_train, y200_train),
                 train_logistic_model(D2000_train, y2000_train)]

quadratic_models = [train_logistic_model(D20_train, y20_train, is_quadratic=True),
                    train_logistic_model(D200_train, y200_train, is_quadratic=True),
                    train_logistic_model(D2000_train, y2000_train, is_quadratic=True)]

# Validation with logistic-linear models
for i, model in enumerate(linear_models, start=1):
    y_score = model.decision_function(D10K_validate)
    fpr, tpr, _ = roc_curve(y10K_validate, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Linear Model (D{20 * (10 ** (i - 1))}_train) AUC = {roc_auc:.2f}')

# Validation with logistic-quadratic models
for i, model in enumerate(quadratic_models, start=1):
    X_val_quad = np.hstack((D10K_validate, D10K_validate[:, 0:1] ** 2, D10K_validate[:, 0:1] * D10K_validate[:, 1:2], D10K_validate[:, 1:2] ** 2))
    y_score = model.decision_function(X_val_quad)
    fpr, tpr, _ = roc_curve(y10K_validate, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Quadratic Model (D{20 * (10 ** (i - 1))}_train) AUC = {roc_auc:.2f}')

# Finalize ROC plot for all models
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Linear and Quadratic Models')
plt.legend(loc='lower right')
plt.grid()
plt.show()
