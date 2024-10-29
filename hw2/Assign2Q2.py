import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[0,:], data[1,:], data[2,:], title='Training Dataset')
    xTrain = data[0:2,:]
    yTrain = data[2,:]
    
    Nvalidate = 1000
    data = generateData(Nvalidate)
    plot3(data[0,:], data[1,:], data[2,:], title='Validation Dataset')
    xValidate = data[0:2,:]
    yValidate = data[2,:]
    
    return xTrain.T, yTrain, xValidate.T, yValidate

def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x, labels = generateDataFromGMM(N, gmmParameters)
    return x

def generateDataFromGMM(N, gmmParameters):
    # Generates N vector samples from the specified mixture of Gaussians
    # Returns samples and their component labels
    # Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    x = np.zeros((n, N))
    labels = np.zeros((1, N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    for l in range(C):
        # Select samples for component l
        if l == 0:
            indl = np.where(u <= thresholds[:,l])
        else:
            indl = np.where((u > thresholds[:,l-1]) & (u <= thresholds[:,l]))
        Nl = len(indl[1])
        if Nl > 0:
            labels[indl] = (l+1)*1
            x[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:,l], covMatrices[:,:,l], Nl))
    return x, labels

def plot3(a, b, c, mark="o", col="b", title='Dataset'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col, alpha=0.6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    plt.show()

def main():
    # Generate Training and Validation Data
    xTrain, yTrain, xValidate, yValidate = hw2q2()
    
    # Construct Polynomial Features (Cubic)
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_train_poly = poly.fit_transform(xTrain)
    X_validate_poly = poly.transform(xValidate)
    
    print(f"Design matrix shape (Training): {X_train_poly.shape}")
    print(f"Design matrix shape (Validation): {X_validate_poly.shape}")
    
    # Fit ML Estimator (OLS)
    ml_model = LinearRegression(fit_intercept=False)  # Bias is included in polynomial features
    ml_model.fit(X_train_poly, yTrain)
    y_pred_ml = ml_model.predict(X_validate_poly)
    mse_ml = mean_squared_error(yValidate, y_pred_ml)
    print(f"ML Estimator MSE on Validation Set: {mse_ml:.4f}")
    
    # Fit MAP Estimator (Ridge Regression) for various gamma values
    # Define range for gamma: from 10^-4 to 10^4 (logarithmically spaced)
    gamma_values = np.logspace(-4, 4, num=100)  # 100 values between 1e-4 and 1e4
    mse_map = []
    optimal_gamma = None
    optimal_mse = np.inf
    optimal_model = None
    
    for gamma in gamma_values:
        alpha = 1.0 / gamma  # Ridge's alpha corresponds to 1/gamma
        ridge_model = Ridge(alpha=alpha, fit_intercept=False)  # Bias is included in polynomial features
        ridge_model.fit(X_train_poly, yTrain)
        y_pred = ridge_model.predict(X_validate_poly)
        mse = mean_squared_error(yValidate, y_pred)
        mse_map.append(mse)
        if mse < optimal_mse:
            optimal_mse = mse
            optimal_gamma = gamma
            optimal_model = ridge_model
    
    print(f"Optimal Gamma (MAP Estimator): {optimal_gamma:.4f} with MSE: {optimal_mse:.4f}")
    print(f"ML Estimator MSE: {mse_ml:.4f}")
    
    # Plot MSE vs Gamma
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_values, mse_map, label='MAP Estimator MSE')
    plt.axhline(y=mse_ml, color='r', linestyle='--', label='ML Estimator MSE')
    plt.xscale('log')
    plt.xlabel('Gamma (γ)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MAP Estimator MSE vs Gamma')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    
    # Plot MSE with Highlight on Optimal Gamma
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_values, mse_map, label='MAP Estimator MSE')
    plt.axhline(y=mse_ml, color='r', linestyle='--', label='ML Estimator MSE')
    plt.axvline(x=optimal_gamma, color='g', linestyle=':', label=f'Optimal γ = {optimal_gamma:.4f}')
    plt.xscale('log')
    plt.xlabel('Gamma (γ)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MAP Estimator MSE vs Gamma with Optimal γ Highlighted')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    
    # Compare Weights of ML and Optimal MAP Estimators
    w_ml = ml_model.coef_
    w_map = optimal_model.coef_
    
    plt.figure(figsize=(12, 6))
    plt.plot(w_ml, 'o-', label='ML Weights')
    plt.plot(w_map, 's-', label=f'MAP Weights (γ={optimal_gamma:.4f})')
    plt.xlabel('Weight Index')
    plt.ylabel('Weight Value')
    plt.title('Comparison of ML and MAP Weights')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Detailed Weight Comparison
    # Plot the absolute difference between ML and MAP weights
    weight_diff = np.abs(w_ml - w_map)
    plt.figure(figsize=(12, 6))
    plt.plot(weight_diff, 'd-', color='purple', label='|ML - MAP| Weights')
    plt.xlabel('Weight Index')
    plt.ylabel('Absolute Difference')
    plt.title('Absolute Difference Between ML and MAP Weights')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
