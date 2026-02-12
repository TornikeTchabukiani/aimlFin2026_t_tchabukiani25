#!/usr/bin/env python3
"""
DDoS Attack Detection and Classification
Identifies attack intervals and classifies attack severity
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import re
from datetime import timedelta

def load_and_prepare_data(filepath, window='1min'):
    """Load logs and prepare for analysis"""
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
                })
    
    df = pd.DataFrame(entries)
    df['time_window'] = df['timestamp'].dt.floor(window)
    
    aggregated = df.groupby('time_window').agg({
        'ip': ['count', 'nunique'],
        'status': lambda x: (x >= 400).sum()
    })
    aggregated.columns = ['total_requests', 'unique_ips', 'error_count']
    aggregated = aggregated.reset_index()
    
    return aggregated

def detect_anomalies(df, threshold=2.0):
    """Detect anomalies using regression and z-scores"""
    df['minutes_elapsed'] = (df['time_window'] - df['time_window'].min()).dt.total_seconds() / 60
    
    X = df[['minutes_elapsed']].values
    y = df['total_requests'].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    df['predicted'] = model.predict(X_poly)
    df['residual'] = df['total_requests'] - df['predicted']
    
    mean = df['residual'].mean()
    std = df['residual'].std()
    df['z_score'] = (df['residual'] - mean) / std
    df['is_anomaly'] = np.abs(df['z_score']) > threshold
    
    return df

def identify_attack_intervals(df, merge_gap_minutes=2):
    """Identify and merge attack intervals"""
    attacks = df[df['is_anomaly']].copy()
    
    if len(attacks) == 0:
        return []
    
    intervals = []
    current_start = attacks.iloc[0]['time_window']
    current_end = current_start
    peak_requests = attacks.iloc[0]['total_requests']
    max_z = abs(attacks.iloc[0]['z_score'])
    peak_errors = attacks.iloc[0]['error_count']
    total_requests = attacks.iloc[0]['total_requests']
    periods = [current_start]
    
    for i in range(1, len(attacks)):
        current_time = attacks.iloc[i]['time_window']
        prev_time = attacks.iloc[i-1]['time_window']
        gap_minutes = (current_time - prev_time).total_seconds() / 60
        
        if gap_minutes <= merge_gap_minutes:
            # Continue current interval
            current_end = current_time
            peak_requests = max(peak_requests, attacks.iloc[i]['total_requests'])
            max_z = max(max_z, abs(attacks.iloc[i]['z_score']))
            peak_errors = max(peak_errors, attacks.iloc[i]['error_count'])
            total_requests += attacks.iloc[i]['total_requests']
            periods.append(current_time)
        else:
            # Save current interval and start new one
            intervals.append({
                'start_time': current_start,
                'end_time': current_end,
                'duration_minutes': (current_end - current_start).total_seconds() / 60,
                'num_periods': len(periods),
                'peak_requests': peak_requests,
                'total_requests': total_requests,
                'max_z_score': max_z,
                'peak_errors': peak_errors,
                'periods': periods.copy()
            })
            
            current_start = current_time
            current_end = current_time
            peak_requests = attacks.iloc[i]['total_requests']
            max_z = abs(attacks.iloc[i]['z_score'])
            peak_errors = attacks.iloc[i]['error_count']
            total_requests = attacks.iloc[i]['total_requests']
            periods = [current_time]
    
    # Add last interval
    intervals.append({
        'start_time': current_start,
        'end_time': current_end,
        'duration_minutes': (current_end - current_start).total_seconds() / 60,
        'num_periods': len(periods),
        'peak_requests': peak_requests,
        'total_requests': total_requests,
        'max_z_score': max_z,
        'peak_errors': peak_errors,
        'periods': periods
    })
    
    return intervals

def classify_attack_severity(interval, baseline_mean):
    """Classify attack severity based on multiple factors"""
    amplitude_ratio = interval['peak_requests'] / baseline_mean
    z_score = interval['max_z_score']
    error_rate = interval['peak_errors'] / interval['peak_requests'] if interval['peak_requests'] > 0 else 0
    
    # Scoring system
    score = 0
    
    # Amplitude factor
    if amplitude_ratio > 10:
        score += 4
    elif amplitude_ratio > 5:
        score += 3
    elif amplitude_ratio > 3:
        score += 2
    elif amplitude_ratio > 2:
        score += 1
    
    # Z-score factor
    if z_score > 4:
        score += 3
    elif z_score > 3:
        score += 2
    elif z_score > 2:
        score += 1
    
    # Error rate factor
    if error_rate > 0.5:
        score += 2
    elif error_rate > 0.3:
        score += 1
    
    # Duration factor
    if interval['duration_minutes'] > 5:
        score += 2
    elif interval['duration_minutes'] > 2:
        score += 1
    
    # Classify
    if score >= 9:
        return 'CRITICAL', score
    elif score >= 6:
        return 'HIGH', score
    elif score >= 4:
        return 'MODERATE', score
    else:
        return 'LOW', score

def analyze_attack_patterns(intervals, df):
    """Analyze patterns across attack intervals"""
    print("\n" + "="*80)
    print("ATTACK PATTERN ANALYSIS")
    print("="*80)
    
    if len(intervals) == 0:
        print("\nNo attacks detected")
        return
    
    # Check for multi-wave pattern
    if len(intervals) > 1:
        print(f"\n⚠️  MULTI-WAVE ATTACK DETECTED - {len(intervals)} distinct waves")
        
        # Analyze gaps between waves
        for i in range(len(intervals) - 1):
            gap = (intervals[i+1]['start_time'] - intervals[i]['end_time']).total_seconds() / 60
            print(f"\nGap between Wave #{i+1} and Wave #{i+2}: {gap:.1f} minutes")
            
            if gap > 0:
                # Analyze gap period
                gap_start = intervals[i]['end_time'] + timedelta(minutes=1)
                gap_end = intervals[i+1]['start_time'] - timedelta(minutes=1)
                gap_data = df[(df['time_window'] > gap_start) & (df['time_window'] < gap_end)]
                
                if len(gap_data) > 0:
                    gap_mean_requests = gap_data['total_requests'].mean()
                    print(f"  Traffic during gap: {gap_mean_requests:.0f} requests/min (average)")
                    print(f"  Interpretation: {'Possible C&C coordination' if gap > 1 else 'Brief pause'}")
        
        # Compare wave intensities
        print(f"\nWave Intensity Comparison:")
        for i, interval in enumerate(intervals, 1):
            print(f"  Wave #{i}: {interval['peak_requests']:,} requests/min (z-score: {interval['max_z_score']:.2f})")
        
        # Check for escalation
        intensities = [i['peak_requests'] for i in intervals]
        if intensities[-1] > intensities[0]:
            increase_pct = ((intensities[-1] / intensities[0]) - 1) * 100
            print(f"\n⚠️  ESCALATING ATTACK: Final wave {increase_pct:.1f}% stronger than initial wave")
    else:
        print(f"\nSingle attack interval detected")
    
    # Analyze overall attack characteristics
    total_duration = sum(i['duration_minutes'] for i in intervals)
    total_attack_requests = sum(i['total_requests'] for i in intervals)
    
    print(f"\nOverall Attack Statistics:")
    print(f"  Total attack duration: {total_duration:.1f} minutes")
    print(f"  Total attack requests: {total_attack_requests:,}")
    print(f"  Average attack intensity: {total_attack_requests / total_duration:.0f} requests/min")

def generate_attack_report(intervals, df, baseline_mean):
    """Generate comprehensive attack report"""
    print("\n" + "="*80)
    print("ATTACK DETECTION REPORT")
    print("="*80)
    
    if len(intervals) == 0:
        print("\n✓ No DDoS attacks detected in the analyzed period")
        return
    
    print(f"\n🚨 {len(intervals)} ATTACK INTERVAL(S) DETECTED\n")
    
    for i, interval in enumerate(intervals, 1):
        severity, score = classify_attack_severity(interval, baseline_mean)
        
        print(f"{'='*80}")
        print(f"ATTACK #{i} - {severity} SEVERITY (Score: {score}/11)")
        print(f"{'='*80}")
        
        print(f"\nTiming:")
        print(f"  Start Time: {interval['start_time']}")
        print(f"  End Time: {interval['end_time']}")
        print(f"  Duration: {interval['duration_minutes']:.1f} minutes")
        print(f"  Number of anomalous periods: {interval['num_periods']}")
        
        print(f"\nTraffic Metrics:")
        print(f"  Peak requests/min: {interval['peak_requests']:,}")
        print(f"  Total requests in attack: {interval['total_requests']:,}")
        print(f"  Baseline (expected): ~{baseline_mean:.0f} requests/min")
        print(f"  Traffic amplification: {(interval['peak_requests'] / baseline_mean):.1f}x normal")
        print(f"  Deviation: +{((interval['peak_requests'] / baseline_mean - 1) * 100):.1f}%")
        
        print(f"\nStatistical Indicators:")
        print(f"  Maximum Z-score: {interval['max_z_score']:.2f}")
        print(f"  Statistical significance: {'Extremely high (p < 0.0001)' if interval['max_z_score'] > 4 else 'High (p < 0.01)' if interval['max_z_score'] > 3 else 'Moderate (p < 0.05)'}")
        
        print(f"\nServer Impact:")
        print(f"  Peak HTTP errors: {interval['peak_errors']} errors/min")
        error_rate = (interval['peak_errors'] / interval['peak_requests'] * 100) if interval['peak_requests'] > 0 else 0
        print(f"  Error rate during peak: {error_rate:.1f}%")
        print(f"  Server status: {'Critical overload' if error_rate > 50 else 'Severe strain' if error_rate > 30 else 'Moderate impact'}")
        
        print(f"\nRecommended Actions:")
        if severity == 'CRITICAL':
            print(f"  • Immediate incident response required")
            print(f"  • Activate DDoS mitigation services")
            print(f"  • Review and block malicious IPs")
            print(f"  • Scale infrastructure immediately")
        elif severity == 'HIGH':
            print(f"  • Urgent investigation needed")
            print(f"  • Enable rate limiting")
            print(f"  • Monitor for continued activity")
        elif severity == 'MODERATE':
            print(f"  • Review attack source IPs")
            print(f"  • Update firewall rules")
            print(f"  • Monitor trends")
        
        print()

def main():
    """Main execution"""
    print("Starting DDoS Attack Detection and Classification...\n")
    
    # Load data
    print("Loading and processing data...")
    df = load_and_prepare_data('server.log')
    print(f"Processed {len(df)} time windows\n")
    
    # Detect anomalies
    print("Detecting anomalies using regression analysis...")
    df = detect_anomalies(df, threshold=2.0)
    baseline_mean = df['predicted'].mean()
    print(f"Baseline traffic: {baseline_mean:.0f} requests/min (mean)\n")
    
    # Identify attack intervals
    print("Identifying attack intervals...")
    intervals = identify_attack_intervals(df, merge_gap_minutes=2)
    print(f"Found {len(intervals)} attack interval(s)\n")
    
    # Analyze patterns
    analyze_attack_patterns(intervals, df)
    
    # Generate report
    generate_attack_report(intervals, df, baseline_mean)
    
    # Save detailed results
    output_file = 'attack_detection_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDetailed analysis saved to: {output_file}")
    
    # Save attack summary
    if len(intervals) > 0:
        summary_file = 'attack_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("DDoS ATTACK SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            for i, interval in enumerate(intervals, 1):
                severity, score = classify_attack_severity(interval, baseline_mean)
                f.write(f"Attack #{i} ({severity}):\n")
                f.write(f"  Time: {interval['start_time']} - {interval['end_time']}\n")
                f.write(f"  Peak: {interval['peak_requests']:,} requests/min\n")
                f.write(f"  Z-score: {interval['max_z_score']:.2f}\n")
                f.write(f"\n")
        print(f"Attack summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("Attack detection and classification complete!")
    print("="*80)

if __name__ == "__main__":
    main()
