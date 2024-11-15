import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import defaultdict
import pandas as pd
import warnings

# Ignore warnings related to GMM convergence and other runtime issues
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------------------------------------
# Step 1: Define True Gaussian Mixture Model (GMM) Parameters
# -----------------------------------------------
# 4 Gaussian components with distinct means and covariance matrices.
# The components are spaced out to reduce overlap.

# Mixing probabilities
true_mixing_probabilities = [0.25, 0.25, 0.25, 0.25]

# Mean vectors
true_mean_vectors = np.array([
    [0, 0],    # Component 1
    [3, 3],    # Component 2
    [6, 0],    # Component 3
    [9, 3]     # Component 4
])

# Covariance matrices
true_covariance_matrices = np.array([
    [[1, 0.3], [0.3, 1]],   # Component 1
    [[1, 0.3], [0.3, 1]],   # Component 2
    [[1, -0.3], [-0.3, 1]], # Component 3
    [[1, 0], [0, 1]]         # Component 4
])

# -------------------------------------------------------------
# Step 2: Function to Generate Synthetic Data from the True GMM
# -------------------------------------------------------------
def generate_synthetic_data(number_of_samples, mixing_probabilities, mean_vectors, covariance_matrices):
    total_components = len(mixing_probabilities)
    generated_samples = []
    
    # Randomly choose which component each sample comes from based on mixing probabilities
    chosen_components = np.random.choice(total_components, size=number_of_samples, p=mixing_probabilities)
    
    for component_index in chosen_components:
        # Generate a sample from the chosen Gaussian component
        sample = np.random.multivariate_normal(
            mean=mean_vectors[component_index],
            cov=covariance_matrices[component_index]
        )
        generated_samples.append(sample)
    
    return np.array(generated_samples)

# -----------------------------------------------------------------------------------
# Step 3: Function to Perform 10-Fold Cross-Validation for GMM Model Order Selection
# -----------------------------------------------------------------------------------
def perform_cross_validation(data_samples, maximum_number_of_components=10, number_of_folds=10, random_seed=None):
    total_samples = data_samples.shape[0]
    # Adjust the number of folds if the dataset is smaller than desired
    actual_number_of_folds = min(number_of_folds, total_samples)
    
    # Initialize K-Fold cross-validation
    k_fold = KFold(n_splits=actual_number_of_folds, shuffle=True, random_state=random_seed)
    
    # Calculate the minimum training set size across all folds
    minimum_training_size = total_samples - actual_number_of_folds + 1
    
    # Set the maximum number of components based on training set size
    adjusted_max_components = min(maximum_number_of_components, minimum_training_size)
    
    # Dictionary to store average log-likelihood scores for each model order
    average_log_likelihood_scores = defaultdict(list)
    
    # Iterate over each possible number of components
    for component_count in range(1, adjusted_max_components + 1):
        log_likelihood_per_fold = []
        
        # Perform cross-validation
        for training_indices, testing_indices in k_fold.split(data_samples):
            training_data = data_samples[training_indices]
            testing_data = data_samples[testing_indices]
            
            try:
                # Initialize and fit the GMM on the training data
                gmm_model = GaussianMixture(
                    n_components=component_count,
                    covariance_type='full',
                    max_iter=500,
                    random_state=random_seed
                )
                gmm_model.fit(training_data)
                
                # Calculate the average log-likelihood of the testing data
                average_log_likelihood = gmm_model.score(testing_data)
                log_likelihood_per_fold.append(average_log_likelihood)
            except ValueError:
                # If the model cannot be fitted, assign negative infinity
                log_likelihood_per_fold.append(-np.inf)
        
        # Compute the mean log-likelihood across all folds for this component count
        mean_log_likelihood = np.mean(log_likelihood_per_fold)
        average_log_likelihood_scores[component_count] = mean_log_likelihood
    
    # Assign negative infinity to model orders that were not evaluated
    for component_count in range(adjusted_max_components + 1, maximum_number_of_components + 1):
        average_log_likelihood_scores[component_count] = -np.inf
    
    # Select the model order with the highest average log-likelihood
    selected_model_order = max(average_log_likelihood_scores, key=average_log_likelihood_scores.get)
    
    return selected_model_order, average_log_likelihood_scores

# -----------------------------------------------------------
# Step 4: Function to Run the Entire Experiment Multiple Times
# -----------------------------------------------------------
def execute_experiment(sample_sizes_list, experiment_repeats=100, maximum_components=10, folds=10, random_seed=None):
    # Initialize a dictionary to store results for each sample size
    experiment_results = {size: [] for size in sample_sizes_list}
    
    # Iterate over each specified sample size
    for sample_size in sample_sizes_list:
        print(f"Processing dataset with {sample_size} samples...")
        
        # Repeat the experiment the specified number of times
        for repeat in range(experiment_repeats):
            # Generate synthetic data
            generated_data = generate_synthetic_data(
                number_of_samples=sample_size,
                mixing_probabilities=true_mixing_probabilities,
                mean_vectors=true_mean_vectors,
                covariance_matrices=true_covariance_matrices
            )
            
            # Perform cross-validation to select the best model order
            best_model_order, _ = perform_cross_validation(
                data_samples=generated_data,
                maximum_number_of_components=maximum_components,
                number_of_folds=folds,
                random_seed=random_seed
            )
            
            # Record the selected model order
            experiment_results[sample_size].append(best_model_order)
    
    return experiment_results

# ------------------------------------------------
# Step 5: Function to Plot the Experiment Results
# ------------------------------------------------
def visualize_results(experiment_results, sample_sizes_list, maximum_components=10):
    for sample_size in sample_sizes_list:
        # Count how many times each model order was selected
        model_order_counts = pd.Series(experiment_results[sample_size]).value_counts().sort_index()
        
        # Prepare data for all possible model orders
        all_model_orders = range(1, maximum_components + 1)
        selection_frequencies = []
        total_experiments = len(experiment_results[sample_size])
        
        for order in all_model_orders:
            count = model_order_counts.get(order, 0)
            frequency_percentage = (count / total_experiments) * 100
            selection_frequencies.append((order, frequency_percentage))
        
        # Unpack the model orders and their corresponding frequencies
        model_orders, frequencies = zip(*selection_frequencies)
        
        # Create a bar chart for the current sample size
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_orders, frequencies, color='skyblue')
        plt.xlabel('Number of GMM Components', fontsize=12)
        plt.ylabel('Selection Frequency (%)', fontsize=12)
        plt.title(f'Model Order Selection Frequency for {sample_size} Samples', fontsize=14)
        plt.xticks(model_orders)
        plt.ylim(0, 100)
        
        # Add frequency labels above each bar
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{freq:.1f}%", ha='center', va='bottom', fontsize=10)
        
        # Add grid lines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# -----------------------------
# Step 6: Main
# -----------------------------
if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)
    
    # Define the list of sample sizes to evaluate
    sample_sizes = [10, 100, 1000]
    
    # Define the number of times to repeat the experiment for each sample size
    number_of_repeats = 100
    
    # Define the maximum number of GMM components to evaluate
    max_gmm_components = 10
    
    # Define the number of folds for cross-validation
    cross_validation_folds = 10
    
    # Execute the experiment
    experiment_outcomes = execute_experiment(
        sample_sizes_list=sample_sizes,
        experiment_repeats=number_of_repeats,
        maximum_components=max_gmm_components,
        folds=cross_validation_folds,
        random_seed=42
    )
    
    # Display the results in table format
    for sample_size in sample_sizes:
        # Count how many times each model order was selected
        counts = pd.Series(experiment_outcomes[sample_size]).value_counts().sort_index()
        
        # Ensure all model orders are represented in the table
        all_orders = range(1, max_gmm_components + 1)
        selection_percentages = []
        total_runs = len(experiment_outcomes[sample_size])
        
        for order in all_orders:
            count = counts.get(order, 0)
            percentage = (count / total_runs) * 100
            selection_percentages.append((order, percentage))
        
        # Create a DataFrame for better visualization
        results_table = pd.DataFrame(selection_percentages, columns=['Model Order', 'Selection Frequency (%)'])
        print(f"\nDataset Size: {sample_size} samples")
        print(results_table.to_string(index=False))
    
    # Plot the results using bar charts
    visualize_results(
        experiment_results=experiment_outcomes,
        sample_sizes_list=sample_sizes,
        maximum_components=max_gmm_components
    )
