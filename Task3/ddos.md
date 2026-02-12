# DDoS Detection Using Regression Analysis - Complete Report

**Author:** Tornike Tchabukiani
**Date:** February 12, 2026  
**Dataset:** [t_tchabukiani25_16928_server.log](./t_tchabukiani25_16928_server.log)

---

## 🚨 DETECTED DDoS ATTACK TIME INTERVALS

**Multi-Wave DDoS Attack Detected - Two Attack Phases:**

### Attack Wave #1 (Moderate Intensity)
- **Start Time:** 2024-03-22 18:40:00+04:00
- **End Time:** 2024-03-22 18:41:00+04:00
- **Duration:** 2 minutes
- **Peak Traffic:** 6,786 requests/minute (254% above baseline)
- **Z-scores:** 2.22 - 2.16
- **HTTP Errors:** 3,771 - 3,806 errors/minute

### Attack Wave #2 (High Intensity)
- **Start Time:** 2024-03-22 18:43:00+04:00
- **End Time:** 2024-03-22 18:44:00+04:00
- **Duration:** 2 minutes
- **Peak Traffic:** 12,292 requests/minute (559% above baseline)
- **Z-scores:** 4.57 - 4.74 (extremely significant)
- **HTTP Errors:** 6,813 - 7,045 errors/minute

**Attack Pattern:** Coordinated multi-wave attack with 2-minute gap between waves, suggesting a sophisticated distributed attack strategy.

**Log File:** [t_tchabukiani25_16928_server.log](./t_tchabukiani25_16928_server.log)

---


## Executive Summary

This report presents a comprehensive analysis of web server traffic logs to detect Distributed Denial of Service (DDoS) attacks using polynomial regression analysis. The methodology employs statistical anomaly detection by modeling normal traffic patterns and identifying significant deviations that indicate attack behavior.

**Key Findings:**
- **Total log entries analyzed:** 84,665 requests over 61 minutes
- **DDoS attack pattern:** Multi-wave coordinated attack with 2 distinct phases
- **Attack Wave #1:** March 22, 2024, 18:40:00 - 18:41:00 (moderate intensity, 254% above baseline)
- **Attack Wave #2:** March 22, 2024, 18:43:00 - 18:44:00 (high intensity, 559% above baseline)
- **Attack severity:** Peak of 12,292 requests/minute (Z-score: 4.74)
- **Attack pattern:** 2-minute gap between waves suggests coordinated sophisticated attack

---

## Table of Contents

### Main Report
1. [Introduction](#1-introduction)
2. [Methodology](#2-methodology)
3. [Data Analysis](#3-data-analysis)
4. [Regression Analysis](#4-regression-analysis)
5. [Results](#5-results)
6. [Visualizations](#6-visualizations)
7. [Code Implementation](#7-code-implementation)
8. [Conclusions](#8-conclusions)
9. [References](#9-references)

### Appendices
- [Appendix A: Quick Start Guide](#appendix-a-quick-start-guide)
  - File Listing
  - Running Programs
  - Methodology Overview
- [Appendix B: Program Code Documentation](#appendix-b-program-code-documentation)
  - Statistical Analysis Program
  - Regression Analysis Program
  - Attack Detection Program
  - Complete Analysis Pipeline
  - Enhanced Visualization Program
  - Usage Instructions
  - Key Algorithms

---

## 1. Introduction

### 1.1 Background

Distributed Denial of Service (DDoS) attacks represent one of the most significant threats to web infrastructure, overwhelming servers with massive volumes of requests from multiple sources. Traditional threshold-based detection methods often fail to adapt to varying traffic patterns and can produce high false-positive rates.

This analysis employs a regression-based approach that models expected traffic behavior and identifies statistical anomalies, providing a more robust and adaptive detection mechanism.

### 1.2 Objective

The primary objectives of this analysis are:
1. Parse and analyze web server access logs
2. Model normal traffic patterns using polynomial regression
3. Detect anomalous traffic spikes indicative of DDoS attacks
4. Identify precise time intervals of attack periods
5. Provide actionable insights for incident response

### 1.3 Data Source

**Log File:** [t_tchabukiani25_16928_server.log](./t_tchabukiani25_16928_server.log)  
**Original URL:** http://max.ge/aiml_final/t_tchabukiani25_16928_server.log

The log file contains Apache-format web server access logs with the following information per request:
- Client IP address
- Timestamp
- HTTP method and endpoint
- Response status code
- Response size
- Referrer URL
- User agent
- Response time (milliseconds)

---

## 2. Methodology

### 2.1 Analytical Approach

The DDoS detection workflow consists of the following steps:

```
[Log Parsing] → [Time Aggregation] → [Regression Modeling] → 
[Residual Analysis] → [Z-Score Calculation] → [Anomaly Detection] → 
[Interval Identification]
```

**Step-by-step process:**
1. **Parse logs:** Extract structured data from raw Apache logs
2. **Aggregate by time:** Group requests into 1-minute windows
3. **Fit regression model:** Apply 2nd-degree polynomial regression to establish traffic baseline
4. **Calculate residuals:** Compute difference between actual and predicted traffic
5. **Compute z-scores:** Standardize residuals for anomaly detection
6. **Detect anomalies:** Flag periods where |z-score| > 2.5
7. **Merge intervals:** Combine adjacent anomalous periods into attack windows

### 2.2 Regression Theory

**Polynomial Regression Model:**

The traffic baseline is modeled using a second-degree polynomial:

$$y = \beta_0 + \beta_1 t + \beta_2 t^2 + \varepsilon$$

Where:
- $y$ = number of requests per minute
- $t$ = time elapsed (minutes)
- $\beta_0, \beta_1, \beta_2$ = model coefficients
- $\varepsilon$ = residual error term

**Residual Analysis:**

The residual for each time window is calculated as:

$$r_i = y_i - \hat{y}_i$$

Where $y_i$ is the actual request count and $\hat{y}_i$ is the predicted value from the regression model.

**Z-Score Normalization:**

To detect anomalies, residuals are standardized into z-scores:

$$z_i = \frac{r_i - \mu_r}{\sigma_r}$$

Where $\mu_r$ is the mean of residuals and $\sigma_r$ is the standard deviation.

**Anomaly Detection Criterion:**

A time window is flagged as anomalous if:

$$|z_i| > \theta$$

Where $\theta = 2.0$ is the detection threshold (representing approximately 95.4% confidence interval).

### 2.3 Implementation Steps

**Data Processing Pipeline:**

1. **Log Parsing**
   - Regular expression pattern matching
   - Timestamp conversion to datetime objects
   - Data type validation and cleaning

2. **Time Window Aggregation (1-minute intervals)**
   - Total request count
   - Unique IP addresses
   - Average response time
   - HTTP error count (status codes ≥ 400)

3. **Feature Engineering**
   - Compute elapsed time from start
   - Transform to polynomial features (degree 2)
   - Normalize time-based features

4. **Model Training**
   - Fit LinearRegression model with polynomial features
   - Generate predictions for all time windows
   - Extract model coefficients for interpretation

5. **Anomaly Detection**
   - Calculate residuals and z-scores
   - Apply threshold-based flagging
   - Merge adjacent anomalous periods

---

## 3. Data Analysis

### 3.1 Dataset Overview

| Metric | Value |
|--------|-------|
| Total log entries | 84,665 |
| Unique IP addresses | 296 |
| Time span | 61.0 minutes |
| Start time | 2024-03-22 18:00:01 (GMT+4) |
| End time | 2024-03-22 19:00:59 (GMT+4) |
| Time windows analyzed | 61 |

### 3.2 Traffic Statistics

| Statistic | Requests per Minute |
|-----------|---------------------|
| Mean | 1,387.95 |
| Median | 830.00 |
| Std Deviation | 2,270.46 |
| Maximum | 12,292 |
| Minimum | 254 |

The high standard deviation (2,270.46) compared to the mean (1,387.95) indicates significant traffic variability, with at least one extreme spike that deviates substantially from normal patterns.

### 3.3 Log Format Example

```
77.111.184.239 - - [2024-03-22 18:00:37+04:00] "POST /usr/login HTTP/1.0" 403 4964 "https://www.morales.info/tag/main/categorieslogin.htm" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A" 4637
```

**Field breakdown:**
- `77.111.184.239` - Client IP address
- `2024-03-22 18:00:37+04:00` - Request timestamp
- `POST /usr/login HTTP/1.0` - HTTP method and endpoint
- `403` - HTTP status code (Forbidden)
- `4964` - Response size in bytes
- `https://www.morales.info/...` - Referrer URL
- `Mozilla/5.0 ...` - User agent string
- `4637` - Response time in milliseconds

---

## 4. Regression Analysis

### 4.1 Model Configuration

**Polynomial Regression Parameters:**
- **Degree:** 2 (quadratic)
- **Model type:** Ordinary Least Squares (OLS)
- **Features:** Time elapsed (minutes) and its square

**Fitted Model:**

$$\text{Requests} = -135.18 + 109.13t - 1.45t^2$$

Where $t$ is the time elapsed in minutes from the start of the observation period.

**Model Coefficients:**
- $\beta_0$ (Intercept) = -135.18
- $\beta_1$ (Linear term) = 109.13
- $\beta_2$ (Quadratic term) = -1.45

The negative quadratic coefficient (-1.45) suggests that traffic follows a concave pattern, with an initial increase that gradually levels off or decreases over time.

### 4.2 Residual Analysis

**Residual Distribution:**

The residuals represent the difference between actual traffic and the regression baseline. Under normal conditions, residuals should follow an approximately normal distribution centered at zero.

**Residual Statistics:**
- Mean residual: ~0 (by construction)
- Standard deviation: 2,270.46 requests/minute
- Maximum positive residual: 10,855 requests/minute (at attack peak)

### 4.3 Z-Score Distribution

**Detection Threshold:** ±2.0 standard deviations

This threshold corresponds to approximately 95.4% confidence interval, meaning:
- Normal traffic: |z-score| ≤ 2.0
- Anomalous traffic (potential attack): |z-score| > 2.0

**Detected Anomalies:**
- Number of anomalous time periods: 4
- Maximum z-score: 4.74 (highly significant deviation)
- Attack pattern: Two distinct waves separated by 2-minute gap

A z-score of 4.74 indicates the traffic spike is 4.74 standard deviations above the expected baseline, representing a probability of less than 0.001% under normal conditions - strongly indicating an attack.

---

## 5. Results

### 5.1 Detected DDoS Attack Intervals

**Multi-Wave Attack Pattern Detected:**

| Wave | Time Period | Duration | Peak Requests/Min | Baseline | Deviation | Max Z-Score | Unique IPs | HTTP Errors |
|------|-------------|----------|-------------------|----------|-----------|-------------|------------|-------------|
| 1 | 18:40:00 - 18:41:00 | 2 min | 6,786 | 1,915 | +254% | 2.22 | 228 | 3,806 |
| - | 18:42:00 | - | Normal traffic (gap) | - | - | - | - | - |
| 2 | 18:43:00 - 18:44:00 | 2 min | 12,292 | 1,882 | +559% | 4.74 | 241 | 7,045 |

**Overall Attack Window:** 2024-03-22 18:40:00 - 18:44:00 (GMT+4)  
**Total Attack Duration:** 4 minutes (with 1-minute gap at 18:42:00)

### 5.2 Attack Characteristics

**Wave #1: Initial Surge (18:40:00 - 18:41:00)**
- **Intensity:** Moderate (254% above baseline)
- **Peak traffic:** 6,786 requests/minute vs 1,915 expected
- **Statistical significance:** Z-scores of 2.22 and 2.16
- **Server impact:** 3,771-3,806 HTTP errors per minute
- **Pattern:** Sustained 2-minute elevated traffic

**Gap Period (18:42:00)**
- **Duration:** 1 minute
- **Traffic:** Returned to near-baseline levels
- **Interpretation:** Possible botnet coordination pause or rate-limiting recovery

**Wave #2: Peak Assault (18:43:00 - 18:44:00)**
- **Intensity:** High (559% above baseline)
- **Peak traffic:** 12,292 requests/minute vs 1,865 expected
- **Statistical significance:** Z-scores of 4.57 and 4.74 (extremely significant, p < 0.0001)
- **Server impact:** 6,813-7,045 HTTP errors per minute
- **Traffic amplification:** 113% increase over Wave #1

### 5.3 Traffic Pattern Interpretation

The detected attack exhibits characteristics typical of **sophisticated multi-wave DDoS attacks**:

1. **Coordinated multi-phase assault:** Two distinct attack waves with controlled pause
2. **Escalating intensity:** Second wave nearly 2x stronger than first wave
3. **High amplitude:** Up to 559% traffic amplification
4. **Distributed sources:** 228-241 unique IPs during attack (similar to normal traffic, suggesting IP spoofing or botnet)
5. **Server degradation:** Massive error rates (56-57% error rate) indicating server overload

**Attack Strategy Analysis:**

This pattern suggests a **coordinated botnet attack** with the following tactical elements:

- **Phase 1 (Wave #1):** Initial probe/softening attack to test defenses and begin resource exhaustion
- **Pause:** Brief respite allowing attackers to assess impact and prepare second wave
- **Phase 2 (Wave #2):** Intensified assault designed to completely overwhelm already-stressed resources

The 2-minute gap between waves is characteristic of:
- Command-and-control (C&C) server coordination
- Distributed botnet synchronization
- Deliberate attack pacing to evade simple rate-limiting defenses

---

## 6. Visualizations

![DDoS Analysis Visualization](<img width="2380" height="1780" alt="ddos_analysis" src="https://github.com/user-attachments/assets/7ad33e0a-0682-4c1f-b0bb-930e9ed065b7" />)


### 6.1 Visualization Interpretation

**Panel 1: Request Rate with Regression Baseline**
- Blue line: Actual HTTP requests per minute
- Orange dashed line: Polynomial regression baseline (degree 2)
- Red shaded areas: Detected attack intervals
- The visualization clearly shows two distinct traffic spikes:
  - First wave at 18:40-18:41 (moderate surge)
  - Second wave at 18:43-18:44 (intense peak)
  - Gap at 18:42 where traffic briefly returns near baseline

**Panel 2: Regression Residuals**
- Blue bars: Normal traffic periods (residuals within threshold)
- Red bars: Anomalous periods (residuals exceeding threshold)
- Zero line: Perfect model prediction
- Four distinct red bars at 18:40, 18:41, 18:43, and 18:44 clearly show the multi-wave attack pattern
- The residuals at 18:43-18:44 are notably larger than 18:40-18:41, confirming escalating intensity

**Panel 3: Z-Score Analysis**
- Green line: Z-score of residuals over time
- Red dashed lines: Detection thresholds (±2.0)
- Red shaded areas: Regions exceeding threshold (attacks)
- Two distinct attack clusters visible:
  - First cluster: z-scores around 2.2 (moderate anomaly)
  - Second cluster: z-scores around 4.6-4.7 (extreme anomaly)
- The gap between clusters at 18:42 shows return to normal z-score range

**Panel 4: Unique IPs and HTTP Errors**
- Green line: Number of unique IP addresses per minute
- Red line: Count of HTTP errors (4xx and 5xx status codes)
- During both attack waves, unique IPs remain relatively stable (suggesting IP spoofing or botnet)
- Error rates spike dramatically during attacks:
  - Wave #1: ~3,800 errors/minute
  - Wave #2: ~7,000 errors/minute
- Error rate escalation confirms increasing server strain across attack phases

---

## 7. Code Implementation

### 7.1 Core Functions

**Log Parsing Function:**

```python
def parse_log_file(filepath):
    """
    Parse Apache-format web server logs
    
    Returns:
        DataFrame with columns: ip, timestamp, method, endpoint, status, size, response_time
    """
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
                    'method': method,
                    'endpoint': endpoint,
                    'status': int(status),
                    'size': int(size),
                    'response_time': int(response_time)
                })
    
    df = pd.DataFrame(entries)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df
```

**Time Window Aggregation:**

```python
def aggregate_by_time_window(df, window='1min'):
    """
    Aggregate requests into time windows
    
    Returns:
        DataFrame with aggregated metrics per time window
    """
    df['time_window'] = df['timestamp'].dt.floor(window)
    
    aggregated = df.groupby('time_window').agg({
        'ip': 'count',  # total requests
        'response_time': 'mean',  # average response time
        'status': lambda x: (x >= 400).sum()  # error count
    }).rename(columns={'ip': 'total_requests', 'response_time': 'avg_response_time', 'status': 'error_count'})
    
    # Add unique IP count
    aggregated['unique_ips'] = df.groupby('time_window')['ip'].nunique()
    
    return aggregated.reset_index()
```

**Regression-Based Detection:**

```python
def detect_ddos_regression(aggregated_df, polynomial_degree=2, z_threshold=2.5):
    """
    Detect DDoS attacks using polynomial regression analysis
    
    Parameters:
        aggregated_df: DataFrame with time-aggregated traffic data
        polynomial_degree: Degree of polynomial for regression (default: 2)
        z_threshold: Z-score threshold for anomaly detection (default: 2.5)
    
    Returns:
        DataFrame with regression results and anomaly flags
    """
    # Create time-based features for regression
    df = aggregated_df.copy()
    df['minutes_elapsed'] = (df['time_window'] - df['time_window'].min()).dt.total_seconds() / 60
    
    # Prepare features for polynomial regression
    X = df[['minutes_elapsed']].values
    y = df['total_requests'].values
    
    # Fit polynomial regression model
    poly_features = PolynomialFeatures(degree=polynomial_degree)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict baseline traffic
    df['predicted_requests'] = model.predict(X_poly)
    
    # Calculate residuals (actual - predicted)
    df['residual'] = df['total_requests'] - df['predicted_requests']
    
    # Calculate z-scores of residuals for anomaly detection
    residual_mean = df['residual'].mean()
    residual_std = df['residual'].std()
    df['z_score'] = (df['residual'] - residual_mean) / residual_std
    
    # Flag anomalies
    df['is_attack'] = np.abs(df['z_score']) > z_threshold
    
    return df
```

**Attack Interval Identification:**

```python
def identify_attack_intervals(df):
    """
    Identify continuous attack intervals by merging adjacent attack periods
    
    Returns:
        List of dictionaries containing attack interval details
    """
    attack_periods = df[df['is_attack']].copy()
    
    if len(attack_periods) == 0:
        return []
    
    intervals = []
    start_time = attack_periods.iloc[0]['time_window']
    end_time = start_time
    max_requests = attack_periods.iloc[0]['total_requests']
    max_z_score = attack_periods.iloc[0]['z_score']
    
    for i in range(1, len(attack_periods)):
        current_time = attack_periods.iloc[i]['time_window']
        prev_time = attack_periods.iloc[i-1]['time_window']
        
        # Check if consecutive (within 2 minutes)
        if (current_time - prev_time).total_seconds() <= 120:
            end_time = current_time
            max_requests = max(max_requests, attack_periods.iloc[i]['total_requests'])
            max_z_score = max(max_z_score, abs(attack_periods.iloc[i]['z_score']))
        else:
            # Save previous interval
            intervals.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration_minutes': (end_time - start_time).total_seconds() / 60,
                'max_requests_per_min': max_requests,
                'max_z_score': max_z_score
            })
            # Start new interval
            start_time = current_time
            end_time = current_time
            max_requests = attack_periods.iloc[i]['total_requests']
            max_z_score = abs(attack_periods.iloc[i]['z_score'])
    
    # Add the last interval
    intervals.append({
        'start_time': start_time,
        'end_time': end_time,
        'duration_minutes': (end_time - start_time).total_seconds() / 60,
        'max_requests_per_min': max_requests,
        'max_z_score': max_z_score
    })
    
    return intervals
```

### 7.2 Main Analysis Workflow

```python
def main():
    """
    Main analysis workflow
    """
    # Parse log file
    df_raw = parse_log_file('t_tchabukiani25_16928_server.log')
    
    # Aggregate by time windows
    df_agg = aggregate_by_time_window(df_raw, window='1min')
    
    # Perform regression analysis
    df_results = detect_ddos_regression(df_agg, polynomial_degree=2, z_threshold=2.5)
    
    # Identify attack intervals
    intervals = identify_attack_intervals(df_results)
    
    # Create visualizations
    create_visualizations(df_results, intervals, output_path='ddos_analysis.png')
    
    # Save results
    df_results.to_csv('ddos_analysis_results.csv', index=False)
    
    return df_results, intervals
```

**Complete source code:** See [ddos_analysis.py](./ddos_analysis.py)

---

## 8. Conclusions

### 8.1 Key Findings

1. **Multi-Wave DDoS Attack Confirmed:** Analysis successfully identified a sophisticated two-wave DDoS attack on March 22, 2024

   - **Wave #1:** 18:40:00-18:41:00 (moderate intensity, 254% above baseline)
   - **Wave #2:** 18:43:00-18:44:00 (high intensity, 559% above baseline)
   - **Gap:** 1-minute pause at 18:42:00 between waves

2. **Attack Severity:** Peak traffic of 12,292 requests/minute with z-score of 4.74, generating 7,045 HTTP errors/minute at maximum intensity

3. **Attack Pattern:** Coordinated multi-phase assault suggesting botnet-orchestrated attack with C&C coordination

4. **Detection Methodology:** Polynomial regression with z-score threshold of 2.0 proved effective for identifying both moderate and severe anomalies

5. **Traffic Characteristics:** 
   - Escalating wave pattern (second wave 81% stronger than first)
   - High error rates (56-57%) indicating complete server overload
   - Distributed sources (228-241 unique IPs) suggesting botnet infrastructure

### 8.2 Limitations

1. **Time Granularity:** 1-minute aggregation may miss sub-minute attack patterns or very short bursts

2. **Single Metric Focus:** Analysis primarily relies on request volume; incorporating additional features (payload size, endpoint diversity, geographic distribution) could improve detection

3. **Model Assumptions:** Polynomial regression assumes smooth traffic patterns; real-world traffic may exhibit more complex behaviors

4. **Threshold Sensitivity:** The z-score threshold of 2.5 is fixed; adaptive thresholding could reduce false positives

5. **Limited Historical Data:** 61-minute observation window may not capture full range of normal traffic variability

### 8.3 Recommendations

**For Incident Response:**
1. Investigate source IPs active during both attack waves (18:40-18:41 and 18:43-18:44)
2. Analyze the 18:42:00 gap period for evidence of C&C communication or coordination
3. Review firewall logs for correlated activity and identify botnet signatures
4. Examine server resource utilization during both attack phases
5. Check for data exfiltration or unauthorized access attempts during high-error periods
6. Correlate unique IPs between waves to identify persistent attack infrastructure

**For Future Detection:**
1. Implement real-time monitoring with automated alerting for multi-wave patterns
2. Deploy adaptive thresholding that can detect escalating attack phases
3. Configure alerts for sustained elevated traffic (>2 minutes) even below critical thresholds
4. Monitor for pause-and-resume attack patterns characteristic of coordinated attacks
5. Establish baseline traffic profiles for different time periods
6. Implement anomaly detection that tracks error rates alongside request volume

**For System Hardening:**
1. Configure DDoS mitigation services with multi-layer protection (CDN, cloud-based)
2. Implement progressive rate limiting that escalates with attack intensity
3. Deploy behavioral analysis to detect coordinated multi-phase attacks
4. Configure auto-scaling with rapid response to sustained traffic increases
5. Implement Web Application Firewall (WAF) with pattern-based blocking
6. Establish incident response playbook specifically for multi-wave attacks
7. Consider implementing CAPTCHA challenges during sustained elevated traffic
8. Deploy connection rate limiting per IP with aggressive throttling during attacks

---

## 9. References

1. **Regression Analysis:**
   - James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
   
2. **Anomaly Detection:**
   - Chandola, V., Banerjee, A., & Kumar, V. (2009). "Anomaly detection: A survey." *ACM Computing Surveys*, 41(3), 1-58.

3. **DDoS Detection Methods:**
   - Zargar, S. T., Joshi, J., & Tipper, D. (2013). "A survey of defense mechanisms against distributed denial of service (DDoS) flooding attacks." *IEEE Communications Surveys & Tutorials*, 15(4), 2046-2069.

4. **Python Libraries:**
   - Pandas: Data manipulation and analysis (https://pandas.pydata.org/)
   - NumPy: Numerical computing (https://numpy.org/)
   - Scikit-learn: Machine learning library (https://scikit-learn.org/)
   - Matplotlib: Data visualization (https://matplotlib.org/)

5. **Statistical Methods:**
   - Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.

---

## Appendix: Reproducibility

To reproduce this analysis:

1. **Clone the repository and navigate to task_3 folder**
2. **Install required dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
3. **Run the analysis script:**
   ```bash
   python ddos_analysis.py
   ```
4. **Output files generated:**
   - `ddos_analysis.png` - Comprehensive visualization
   - `ddos_analysis_results.csv` - Detailed time-series results

**System Requirements:**
- Python 3.7 or higher
- 2GB RAM minimum
- ~50MB disk space for data and outputs

---

**Report End**

---

# APPENDIX A: Quick Start Guide

1. **README.md** - This file (quick start guide)
2. **ddos.md** - Comprehensive analysis report (25KB, 9 sections)
3. **PROGRAMS.md** - Detailed program code documentation

### 💻 Python Programs (5 modules)
1. **ddos_analysis.py** - Complete integrated DDoS analysis pipeline (14KB)
2. **statistical_analysis.py** - Statistical data extraction (11KB)
3. **regression_analysis.py** - Regression modeling and diagnostics (11KB)
4. **attack_detection.py** - Attack detection and severity classification (13KB)
5. **enhanced_visualization.py** - Publication-quality 300 DPI visualization (15KB)

### 📊 Visualizations
1. **enhanced_ddos_analysis.png** - 8-panel publication-quality dashboard (1.6MB, 300 DPI) ⭐
2. **ddos_analysis.png** - 4-panel DDoS analysis visualization (298KB, 300 DPI)
3. **statistical_analysis.png** - 6-panel statistical analysis (249KB)
4. **regression_analysis.png** - 6-panel regression diagnostics (272KB)

### 📝 Output Files
1. **attack_summary.txt** - Brief attack summary
2. **t_tchabukiani25_16928_server.log** - Original log file (84,665 entries, 20MB)

---

## 🚨 Attack Detection Results

**Multi-Wave DDoS Attack Detected:**

**Wave #1 (Moderate):**
- **Time:** March 22, 2024, 18:40:00 - 18:41:00 (GMT+4)
- **Severity:** 6,786 requests/minute (254% above baseline)
- **Z-Score:** 2.16 - 2.22

**Wave #2 (High Intensity):**
- **Time:** March 22, 2024, 18:43:00 - 18:44:00 (GMT+4)
- **Severity:** 12,292 requests/minute (559% above baseline)
- **Z-Score:** 4.57 - 4.74 (extremely significant)

**Attack Pattern:** Coordinated two-wave attack with 1-minute gap, suggesting sophisticated botnet-orchestrated assault.

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Run Individual Programs

**1. Complete DDoS Analysis (Recommended)**
```bash
python ddos_analysis.py
```
Output: `ddos_analysis.png`, `ddos_analysis_results.csv`

**2. Statistical Analysis**
```bash
python statistical_analysis.py
```
Output: `statistical_analysis.png`, comprehensive console report

**3. Regression Analysis**
```bash
python regression_analysis.py
```
Output: `regression_analysis.png`, `regression_results.csv`

**4. Attack Detection & Classification**
```bash
python attack_detection.py
```
Output: `attack_detection_results.csv`, `attack_summary.txt`

**5. Enhanced Publication-Quality Visualization (Recommended for Reports)**
```bash
python enhanced_visualization.py
```
Output: `enhanced_ddos_analysis.png` (300 DPI, 8-panel dashboard with detailed annotations)

---

1. Parse Apache-format web server logs
2. Aggregate requests into 1-minute time windows
3. Fit 2nd-degree polynomial regression to model baseline traffic
4. Calculate residuals and z-scores for anomaly detection
5. Identify attack intervals where |z-score| > 2.0
6. Generate comprehensive visualizations and reports

## Files Description

### ddos.md
Complete report including:
- Executive summary
- Detailed methodology with mathematical formulas
- Data analysis and statistics
- Regression model details
- Attack detection results
- Visualizations and interpretations
- Source code documentation
- Conclusions and recommendations

### ddos_analysis.py
Python implementation featuring:
- Log parsing with regex pattern matching
- Time-series aggregation
- Polynomial regression modeling
- Z-score based anomaly detection
- Attack interval identification
- Multi-panel visualization generation

### ddos_analysis.png
Four-panel visualization showing:
1. Request rate with regression baseline
2. Residuals distribution
3. Z-score time series
4. Unique IPs and HTTP errors

### t_tchabukiani25_16928_server.log
Original Apache-format log file containing:
- 84,665 HTTP requests
- 61-minute observation period
- 296 unique IP addresses
- Full request metadata

## Author

Tornike Tchabukiani (t_tchabukiani25_16928)

## Date

February 12, 2026

---

# APPENDIX B: Program Code Documentation

This document describes all the program codes developed for DDoS detection and statistical analysis of web server logs.

---

## Table of Contents

1. [statistical_analysis.py](#1-statistical_analysispy) - Statistical Data Extraction
2. [regression_analysis.py](#2-regression_analysispy) - Regression Modeling
3. [attack_detection.py](#3-attack_detectionpy) - Attack Detection & Classification
4. [ddos_analysis.py](#4-ddos_analysispy) - Complete Integrated Analysis
5. [enhanced_visualization.py](#5-enhanced_visualizationpy) - Publication-Quality Visualization
6. [Usage Instructions](#usage-instructions)

---

## 1. statistical_analysis.py

**Purpose:** Extract comprehensive statistical data from web server log files

**Features:**
- Parse Apache-format log files
- Compute basic statistics (request counts, response times, data transfer)
- Analyze HTTP methods and status code distributions
- Identify top endpoints and IP addresses
- Detect suspicious IP patterns
- Analyze temporal patterns (hourly traffic)
- Examine user agent distributions
- Generate 6-panel visualization of statistical metrics

**Key Functions:**

```python
parse_log_file(filepath)
    # Parse Apache logs into structured DataFrame
    
compute_basic_statistics(df)
    # Calculate dataset overview and request statistics
    
analyze_http_methods(df)
    # Distribution of GET, POST, PUT, DELETE requests
    
analyze_status_codes(df)
    # Success (2xx), Redirect (3xx), Client Error (4xx), Server Error (5xx)
    
analyze_endpoints(df)
    # Most frequently accessed endpoints
    
analyze_ip_addresses(df)
    # Top IPs, requests per IP statistics, suspicious IP detection
    
analyze_temporal_patterns(df)
    # Hourly request distribution
    
analyze_user_agents(df)
    # Browser distribution and user agent analysis
    
create_statistical_visualizations(df)
    # Generate comprehensive statistical plots
```

**Output:**
- Console: Detailed statistical report
- File: `statistical_analysis.png` (6 visualization panels)

**Key Statistics Generated:**
- Total requests: 84,665
- Unique IPs: 296
- Duration: 61 minutes
- Mean response time: 2,497.63 ms
- Status code distribution: 14% success, 29% redirects, 29% client errors, 28% server errors

---

## 2. regression_analysis.py

**Purpose:** Perform polynomial regression analysis for baseline modeling

**Features:**
- Compare polynomial degrees (1-4) using AIC criterion
- Fit optimal regression model to traffic data
- Comprehensive residual analysis
- Z-score computation for anomaly detection
- Normality testing (Shapiro-Wilk test)
- Generate 6-panel regression diagnostic plots

**Key Functions:**

```python
parse_and_aggregate(filepath, window='1min')
    # Parse logs and aggregate into time windows
    
fit_polynomial_regression(df, degree=2)
    # Fit polynomial regression model
    # Returns: model, predictions, R², MSE, RMSE, MAE
    
analyze_residuals(y_true, y_pred)
    # Residual statistics and normality testing
    
compute_z_scores(residuals, threshold=2.0)
    # Calculate z-scores and identify anomalies
    
compare_polynomial_degrees(df, degrees=[1,2,3,4])
    # Compare model fit using AIC criterion
    
create_regression_visualizations(...)
    # 6-panel diagnostic plot:
    #   1. Actual vs Predicted
    #   2. Residuals vs Fitted
    #   3. Q-Q Plot (normality)
    #   4. Residual Histogram
    #   5. Z-scores over time
    #   6. Scale-Location plot
```

**Mathematical Model:**

$$y = \beta_0 + \beta_1 t + \beta_2 t^2 + \varepsilon$$

Where:
- $y$ = requests per minute
- $t$ = time elapsed (minutes)
- Fitted coefficients: β₀=-135.18, β₁=109.13, β₂=-1.45

**Output:**
- Console: Regression statistics and model comparison
- File: `regression_analysis.png` (6 diagnostic plots)
- File: `regression_results.csv` (detailed time-series results)

**Key Metrics:**
- R² Score: 0.0622
- RMSE: 2,180.66
- Best polynomial degree: 3 (lowest AIC)
- Anomalies detected: 4 time periods (6.56% of data)

---

## 3. attack_detection.py

**Purpose:** Detect and classify DDoS attack intervals with severity scoring

**Features:**
- Regression-based anomaly detection
- Attack interval identification and merging
- Multi-wave pattern detection
- Attack severity classification (LOW/MODERATE/HIGH/CRITICAL)
- Gap analysis between attack waves
- Comprehensive attack characterization

**Key Functions:**

```python
load_and_prepare_data(filepath, window='1min')
    # Load logs and prepare for analysis
    
detect_anomalies(df, threshold=2.0)
    # Regression + z-score anomaly detection
    
identify_attack_intervals(df, merge_gap_minutes=2)
    # Merge adjacent anomalous periods into attack intervals
    
classify_attack_severity(interval, baseline_mean)
    # Classify as LOW/MODERATE/HIGH/CRITICAL
    # Scoring based on:
    #   - Amplitude ratio (traffic amplification)
    #   - Z-score magnitude
    #   - Error rate
    #   - Attack duration
    
analyze_attack_patterns(intervals, df)
    # Multi-wave detection
    # Gap analysis
    # Escalation detection
    
generate_attack_report(intervals, df, baseline_mean)
    # Comprehensive attack report
```

**Severity Classification Algorithm:**

```
Score = amplitude_factor + z_score_factor + error_rate_factor + duration_factor

CRITICAL: score ≥ 9
HIGH:     score ≥ 6
MODERATE: score ≥ 4
LOW:      score < 4
```

**Output:**
- Console: Attack detection report with severity classification
- File: `attack_detection_results.csv` (detailed analysis)
- File: `attack_summary.txt` (attack summary)

**Detected Attacks:**
- Attack #1: CRITICAL severity (score 9/11)
  - Duration: 4.0 minutes (4 anomalous periods)
  - Peak: 12,292 requests/min (8.9x amplification)
  - Z-score: 4.74
  - Error rate: 57.3% (critical overload)

---

## 4. ddos_analysis.py

**Purpose:** Complete integrated DDoS analysis pipeline

**Features:**
- All-in-one analysis tool combining statistical and regression methods
- 4-panel comprehensive visualization
- Attack interval detection and merging
- CSV export of detailed results

**Key Functions:**

```python
parse_log_file(filepath)
    # Parse Apache-format logs
    
aggregate_by_time_window(df, window='1min')
    # Aggregate to 1-minute intervals
    
detect_ddos_regression(df, polynomial_degree=2, z_threshold=2.0)
    # Full regression pipeline:
    #   - Fit polynomial model
    #   - Calculate residuals
    #   - Compute z-scores
    #   - Flag anomalies
    
identify_attack_intervals(df)
    # Merge adjacent attacks
    
create_visualizations(df, intervals)
    # 4-panel plot:
    #   1. Request rate + regression baseline
    #   2. Residuals bar chart
    #   3. Z-scores with thresholds
    #   4. Unique IPs + errors
    
print_statistics(df, raw_df, intervals)
    # Comprehensive console report
```

**Output:**
- Console: Complete analysis report
- File: `ddos_analysis.png` (4-panel visualization)
- File: `ddos_analysis_results.csv` (time-series data)

---

## 5. enhanced_visualization.py

**Purpose:** Generate publication-quality enhanced visualizations (300 DPI)

**Features:**
- Ultra high-resolution output (300 DPI, 20"×14")
- 8-panel comprehensive analysis dashboard
- Detailed attack wave annotations with arrows and callouts
- Enhanced color scheme and typography
- Response time analysis during attacks
- Traffic distribution comparison (normal vs attack)
- Error rate percentage tracking
- Multi-wave attack timeline summary panel

**Visualization Panels:**

1. **Main Traffic Analysis** - Request rate with regression baseline, annotated attack waves
2. **Residuals Distribution** - Color-coded residual bars with ±2σ threshold lines
3. **Z-Score Analysis** - Statistical anomaly detection with peak annotations
4. **Unique IPs & Errors** - Dual-axis plot showing source distribution
5. **Response Time** - Mean response time with standard deviation bands
6. **Traffic Distribution** - Histogram comparing normal vs attack traffic
7. **Error Rate %** - HTTP error percentage with critical threshold
8. **Attack Timeline** - Comprehensive multi-wave attack summary in monospace format

**Output:**
- File: `enhanced_ddos_analysis.png` (1.6MB, 300 DPI, publication-ready)

**Special Features:**
- Detailed wave-by-wave comparison
- Coordinated C&C pause detection
- Server overload indicators
- Escalation pattern visualization
- Professional annotation system

---

## Usage Instructions

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Running the Programs

**1. Statistical Analysis**
```bash
python statistical_analysis.py
```
Generates: `statistical_analysis.png`

**2. Regression Analysis**
```bash
python regression_analysis.py
```
Generates: `regression_analysis.png`, `regression_results.csv`

**3. Attack Detection**
```bash
python attack_detection.py
```
Generates: `attack_detection_results.csv`, `attack_summary.txt`

**4. Complete Analysis (Recommended)**
```bash
python ddos_analysis.py
```
Generates: `ddos_analysis.png`, `ddos_analysis_results.csv`

**5. Enhanced High-Resolution Visualization (Publication Quality)**
```bash
python enhanced_visualization.py
```
Generates: `enhanced_ddos_analysis.png` (300 DPI, 1.6MB, 8-panel dashboard)

### Input Requirements

All programs expect a log file named `server.log` or `t_tchabukiani25_16928_server.log` in the same directory, with Apache-format entries:

```
IP - - [TIMESTAMP] "METHOD ENDPOINT HTTP/x.x" STATUS SIZE "REFERRER" "USER_AGENT" RESPONSE_TIME
```

### Customization

**Change time window aggregation:**
```python
df = aggregate_by_time_window(df, window='5min')  # 5-minute windows
```

**Adjust anomaly threshold:**
```python
df = detect_ddos_regression(df, z_threshold=1.5)  # More sensitive
```

**Change polynomial degree:**
```python
df = detect_ddos_regression(df, polynomial_degree=3)  # Cubic model
```

---

## Program Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    Log File (server.log)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬──────────────┐
        │            │            │              │
        ▼            ▼            ▼              ▼
┌───────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐
│Statistical│  │ Regression │  │ Attack   │  │  Complete  │
│ Analysis  │  │  Analysis  │  │ Detection│  │  Pipeline  │
└─────┬─────┘  └─────┬──────┘  └────┬─────┘  └─────┬──────┘
      │              │               │              │
      ▼              ▼               ▼              ▼
  stats.png    regression.png   attack_    ddos_analysis.png
                                summary.txt
```

---

## Key Algorithms

### 1. Polynomial Regression

```python
# Feature transformation
X_poly = [1, t, t²]

# Model fitting
β = (X'X)⁻¹X'y

# Prediction
ŷ = β₀ + β₁t + β₂t²
```

### 2. Z-Score Anomaly Detection

```python
# Residual calculation
r = y - ŷ

# Z-score normalization
z = (r - μᵣ) / σᵣ

# Anomaly flagging
is_anomaly = |z| > threshold
```

### 3. Attack Interval Merging

```python
# Merge adjacent anomalies
if (current_time - prev_time) ≤ gap_threshold:
    # Continue current interval
else:
    # Start new interval
```

---

## Analysis Results Summary

| Metric | Value |
|--------|-------|
| Total Requests | 84,665 |
| Time Span | 61 minutes |
| Baseline Traffic | 1,388 req/min |
| Anomalous Periods | 4 |
| Attack Intervals | 1 (multi-wave) |
| Peak Traffic | 12,292 req/min |
| Traffic Amplification | 8.9x |
| Max Z-Score | 4.74 |
| Attack Severity | CRITICAL |

---

## Author

Tornike Tchabukiani (t_tchabukiani25_16928)

## Date

February 12, 2026
