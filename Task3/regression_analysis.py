#!/usr/bin/env python3
"""
Regression Analysis for DDoS Detection
Implements polynomial regression with comprehensive statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import re

sns.set_style("whitegrid")

def parse_and_aggregate(filepath, window='1min'):
    """Parse logs and aggregate by time window"""
    log_pattern = r'(\d+\.\d+\.\d+\.\d+)\s+-\s+-\s+\[([^\]]+)\]\s+"(\w+)\s+([^\s]+)\s+HTTP/[^"]+"\s+(\d+)\s+(\d+)\s+"([^"]*)"\s+"([^"]*)"\s+(\d+)'
    
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(log_pattern, line.strip())
            if match:
                ip, timestamp, method, endpoint, status, size, referrer, user_agent, response_time = match.groups()
                entries.append({
                    'ip': ip,
                    'timestamp': pd.to_datetime(timestamp),
                    'status': int(status),
                    'response_time': int(response_time)
                })
    
    df = pd.DataFrame(entries)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Aggregate
    df['time_window'] = df['timestamp'].dt.floor(window)
    aggregated = df.groupby('time_window').agg({
        'ip': ['count', 'nunique'],
        'response_time': 'mean',
        'status': lambda x: (x >= 400).sum()
    })
    
    aggregated.columns = ['total_requests', 'unique_ips', 'avg_response_time', 'error_count']
    aggregated = aggregated.reset_index()
    
    return aggregated

def fit_polynomial_regression(df, degree=2):
    """
    Fit polynomial regression model
    
    Returns:
        model, predictions, metrics
    """
    print("="*80)
    print(f"POLYNOMIAL REGRESSION ANALYSIS (Degree = {degree})")
    print("="*80)
    
    # Prepare features
    df['minutes_elapsed'] = (df['time_window'] - df['time_window'].min()).dt.total_seconds() / 60
    X = df[['minutes_elapsed']].values
    y = df['total_requests'].values
    
    # Transform features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Generate predictions
    y_pred = model.predict(X_poly)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\nModel Coefficients:")
    print(f"  Intercept (β₀): {model.intercept_:.4f}")
    for i, coef in enumerate(model.coef_[1:], 1):
        print(f"  Coefficient β{i}: {coef:.4f}")
    
    print(f"\nModel Performance Metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    
    # Regression equation
    if degree == 2:
        print(f"\nRegression Equation:")
        print(f"  y = {model.intercept_:.2f} + {model.coef_[1]:.2f}t + {model.coef_[2]:.2f}t²")
    
    return model, poly, y_pred, {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}

def analyze_residuals(y_true, y_pred):
    """Perform comprehensive residual analysis"""
    print("\n" + "="*80)
    print("RESIDUAL ANALYSIS")
    print("="*80)
    
    residuals = y_true - y_pred
    
    print(f"\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.4f}")
    print(f"  Std Dev: {residuals.std():.4f}")
    print(f"  Min: {residuals.min():.4f}")
    print(f"  Max: {residuals.max():.4f}")
    print(f"  Q1 (25%): {np.percentile(residuals, 25):.4f}")
    print(f"  Median (50%): {np.percentile(residuals, 50):.4f}")
    print(f"  Q3 (75%): {np.percentile(residuals, 75):.4f}")
    
    # Test for normality (Shapiro-Wilk test)
    statistic, p_value = stats.shapiro(residuals)
    print(f"\nNormality Test (Shapiro-Wilk):")
    print(f"  Test Statistic: {statistic:.4f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value > 0.05:
        print(f"  Result: Residuals appear normally distributed (p > 0.05)")
    else:
        print(f"  Result: Residuals may not be normally distributed (p ≤ 0.05)")
    
    return residuals

def compute_z_scores(residuals, threshold=2.0):
    """Compute z-scores and identify anomalies"""
    print("\n" + "="*80)
    print("Z-SCORE ANALYSIS")
    print("="*80)
    
    mean = residuals.mean()
    std = residuals.std()
    z_scores = (residuals - mean) / std
    
    print(f"\nZ-Score Statistics:")
    print(f"  Mean: {z_scores.mean():.4f}")
    print(f"  Std Dev: {z_scores.std():.4f}")
    print(f"  Min: {z_scores.min():.4f}")
    print(f"  Max: {z_scores.max():.4f}")
    
    print(f"\nAnomaly Detection (Threshold = ±{threshold}):")
    anomalies = np.abs(z_scores) > threshold
    n_anomalies = anomalies.sum()
    anomaly_pct = (n_anomalies / len(z_scores)) * 100
    
    print(f"  Number of anomalies: {n_anomalies}")
    print(f"  Percentage of data: {anomaly_pct:.2f}%")
    
    # Show anomalous periods
    if n_anomalies > 0:
        print(f"\n  Anomalous periods:")
        anomaly_indices = np.where(anomalies)[0]
        for idx in anomaly_indices:
            print(f"    Index {idx}: z-score = {z_scores[idx]:.2f}, residual = {residuals[idx]:.2f}")
    
    return z_scores, anomalies

def compare_polynomial_degrees(df, degrees=[1, 2, 3, 4]):
    """Compare different polynomial degrees"""
    print("\n" + "="*80)
    print("POLYNOMIAL DEGREE COMPARISON")
    print("="*80)
    
    df['minutes_elapsed'] = (df['time_window'] - df['time_window'].min()).dt.total_seconds() / 60
    X = df[['minutes_elapsed']].values
    y = df['total_requests'].values
    
    results = []
    
    print(f"\n{'Degree':<10} {'R²':<12} {'RMSE':<12} {'MAE':<12} {'AIC':<12}")
    print("-" * 60)
    
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Calculate AIC (Akaike Information Criterion)
        n = len(y)
        k = degree + 1  # number of parameters
        rss = np.sum((y - y_pred) ** 2)
        aic = n * np.log(rss / n) + 2 * k
        
        print(f"{degree:<10} {r2:<12.4f} {rmse:<12.2f} {mae:<12.2f} {aic:<12.2f}")
        
        results.append({
            'degree': degree,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'aic': aic
        })
    
    print("\nRecommendation:")
    best_aic = min(results, key=lambda x: x['aic'])
    print(f"  Degree {best_aic['degree']} has the lowest AIC ({best_aic['aic']:.2f})")
    print(f"  This suggests it provides the best balance between fit and complexity")
    
    return results

def create_regression_visualizations(df, y_pred, residuals, z_scores, anomalies, 
                                     output_path='regression_analysis.png'):
    """Create detailed regression analysis visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(df['minutes_elapsed'], df['total_requests'], alpha=0.6, label='Actual', s=50)
    ax1.plot(df['minutes_elapsed'], y_pred, 'r-', linewidth=2, label='Predicted (Regression)')
    ax1.set_xlabel('Time Elapsed (minutes)')
    ax1.set_ylabel('Requests per Minute')
    ax1.set_title('Actual vs Predicted Request Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs Fitted
    ax2 = axes[0, 1]
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Fitted Values')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot (normality check)
    ax3 = axes[0, 2]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Normal Q-Q Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residual Histogram
    ax4 = axes[1, 0]
    ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(residuals.mean(), color='r', linestyle='--', 
                label=f'Mean: {residuals.mean():.2f}')
    ax4.set_xlabel('Residual Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Z-scores over time
    ax5 = axes[1, 1]
    colors = ['red' if a else 'blue' for a in anomalies]
    ax5.scatter(df['minutes_elapsed'], z_scores, c=colors, alpha=0.6, s=50)
    ax5.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Threshold (±2.0)')
    ax5.axhline(y=-2.0, color='r', linestyle='--', linewidth=2)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Time Elapsed (minutes)')
    ax5.set_ylabel('Z-Score')
    ax5.set_title('Z-Scores of Residuals (Red = Anomaly)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Scale-Location plot
    ax6 = axes[1, 2]
    standardized_residuals = np.sqrt(np.abs(residuals / residuals.std()))
    ax6.scatter(y_pred, standardized_residuals, alpha=0.6, s=50)
    ax6.set_xlabel('Fitted Values')
    ax6.set_ylabel('√|Standardized Residuals|')
    ax6.set_title('Scale-Location Plot')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nRegression visualizations saved to: {output_path}")

def main():
    """Main execution"""
    print("Starting Regression Analysis for DDoS Detection...\n")
    
    # Load and aggregate data
    print("Loading and aggregating data...")
    df = parse_and_aggregate('server.log', window='1min')
    print(f"Created {len(df)} time windows\n")
    
    # Compare polynomial degrees
    compare_polynomial_degrees(df, degrees=[1, 2, 3, 4])
    
    # Fit polynomial regression (degree 2)
    model, poly, y_pred, metrics = fit_polynomial_regression(df, degree=2)
    
    # Analyze residuals
    residuals = analyze_residuals(df['total_requests'].values, y_pred)
    
    # Compute z-scores
    z_scores, anomalies = compute_z_scores(residuals, threshold=2.0)
    
    # Add results to dataframe
    df['predicted_requests'] = y_pred
    df['residual'] = residuals
    df['z_score'] = z_scores
    df['is_anomaly'] = anomalies
    
    # Create visualizations
    create_regression_visualizations(df, y_pred, residuals, z_scores, anomalies)
    
    # Save results
    output_file = 'regression_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("Regression analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
