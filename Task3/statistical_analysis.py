#!/usr/bin/env python3
"""
Statistical Analysis of Web Server Logs
Extracts comprehensive statistical data from Apache-format log files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter

def parse_log_file(filepath):
    """Parse Apache-format web server logs"""
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
                    'referrer': referrer,
                    'user_agent': user_agent,
                    'response_time': int(response_time)
                })
    
    return pd.DataFrame(entries)

def compute_basic_statistics(df):
    """Compute basic statistical metrics"""
    print("="*80)
    print("BASIC STATISTICS")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"  Total requests: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60:.1f} minutes")
    
    print(f"\nUnique Values:")
    print(f"  Unique IP addresses: {df['ip'].nunique():,}")
    print(f"  Unique endpoints: {df['endpoint'].nunique()}")
    print(f"  Unique user agents: {df['user_agent'].nunique()}")
    
    print(f"\nRequest Statistics:")
    print(f"  Mean response time: {df['response_time'].mean():.2f} ms")
    print(f"  Median response time: {df['response_time'].median():.2f} ms")
    print(f"  Std dev response time: {df['response_time'].std():.2f} ms")
    print(f"  Max response time: {df['response_time'].max()} ms")
    print(f"  Min response time: {df['response_time'].min()} ms")
    
    print(f"\nResponse Size Statistics:")
    print(f"  Mean size: {df['size'].mean():.2f} bytes")
    print(f"  Median size: {df['size'].median():.2f} bytes")
    print(f"  Total data transferred: {df['size'].sum() / (1024**2):.2f} MB")

def analyze_http_methods(df):
    """Analyze HTTP method distribution"""
    print("\n" + "="*80)
    print("HTTP METHOD DISTRIBUTION")
    print("="*80)
    
    method_counts = df['method'].value_counts()
    print("\nMethod counts:")
    for method, count in method_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {method:<10} {count:>8,} ({pct:>5.2f}%)")

def analyze_status_codes(df):
    """Analyze HTTP status code distribution"""
    print("\n" + "="*80)
    print("HTTP STATUS CODE DISTRIBUTION")
    print("="*80)
    
    status_counts = df['status'].value_counts().sort_index()
    
    # Categorize status codes
    success = df[df['status'] < 300]
    redirects = df[(df['status'] >= 300) & (df['status'] < 400)]
    client_errors = df[(df['status'] >= 400) & (df['status'] < 500)]
    server_errors = df[df['status'] >= 500]
    
    print(f"\nStatus Code Categories:")
    print(f"  2xx (Success):       {len(success):>8,} ({len(success)/len(df)*100:>5.2f}%)")
    print(f"  3xx (Redirect):      {len(redirects):>8,} ({len(redirects)/len(df)*100:>5.2f}%)")
    print(f"  4xx (Client Error):  {len(client_errors):>8,} ({len(client_errors)/len(df)*100:>5.2f}%)")
    print(f"  5xx (Server Error):  {len(server_errors):>8,} ({len(server_errors)/len(df)*100:>5.2f}%)")
    
    print(f"\nTop 10 Status Codes:")
    for status, count in status_counts.head(10).items():
        pct = (count / len(df)) * 100
        print(f"  {status:>3} {count:>8,} ({pct:>5.2f}%)")

def analyze_endpoints(df):
    """Analyze most frequently accessed endpoints"""
    print("\n" + "="*80)
    print("ENDPOINT ANALYSIS")
    print("="*80)
    
    endpoint_counts = df['endpoint'].value_counts()
    print(f"\nTop 15 Most Requested Endpoints:")
    for endpoint, count in endpoint_counts.head(15).items():
        pct = (count / len(df)) * 100
        print(f"  {endpoint:<40} {count:>8,} ({pct:>5.2f}%)")

def analyze_ip_addresses(df):
    """Analyze IP address patterns"""
    print("\n" + "="*80)
    print("IP ADDRESS ANALYSIS")
    print("="*80)
    
    ip_counts = df['ip'].value_counts()
    
    print(f"\nTop 20 Most Active IP Addresses:")
    for ip, count in ip_counts.head(20).items():
        pct = (count / len(df)) * 100
        print(f"  {ip:<20} {count:>8,} requests ({pct:>5.2f}%)")
    
    # Analyze IP request distribution
    requests_per_ip = ip_counts.values
    print(f"\nRequests per IP Statistics:")
    print(f"  Mean requests/IP: {requests_per_ip.mean():.2f}")
    print(f"  Median requests/IP: {np.median(requests_per_ip):.2f}")
    print(f"  Max requests from single IP: {requests_per_ip.max()}")
    print(f"  Min requests from single IP: {requests_per_ip.min()}")
    
    # Identify potential suspicious IPs (high request count)
    suspicious_threshold = requests_per_ip.mean() + 2 * requests_per_ip.std()
    suspicious_ips = ip_counts[ip_counts > suspicious_threshold]
    
    if len(suspicious_ips) > 0:
        print(f"\nPotentially Suspicious IPs (>{suspicious_threshold:.0f} requests):")
        for ip, count in suspicious_ips.items():
            print(f"  {ip:<20} {count:>8,} requests")

def analyze_temporal_patterns(df):
    """Analyze time-based patterns"""
    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS")
    print("="*80)
    
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    hourly_requests = df.groupby('hour').size()
    
    print(f"\nRequests by Hour:")
    for hour, count in hourly_requests.items():
        bar = '█' * int(count / 1000)
        print(f"  {hour:02d}:00  {count:>6,} {bar}")

def analyze_user_agents(df):
    """Analyze user agent distribution"""
    print("\n" + "="*80)
    print("USER AGENT ANALYSIS")
    print("="*80)
    
    ua_counts = df['user_agent'].value_counts()
    
    print(f"\nTop 10 User Agents:")
    for ua, count in ua_counts.head(10).items():
        pct = (count / len(df)) * 100
        ua_short = ua[:60] + "..." if len(ua) > 60 else ua
        print(f"  {ua_short:<63} {count:>6,} ({pct:>5.2f}%)")
    
    # Identify browser types
    browsers = {
        'Chrome': df['user_agent'].str.contains('Chrome', case=False).sum(),
        'Firefox': df['user_agent'].str.contains('Firefox', case=False).sum(),
        'Safari': df['user_agent'].str.contains('Safari', case=False).sum(),
        'Edge': df['user_agent'].str.contains('Edg', case=False).sum(),
        'Opera': df['user_agent'].str.contains('OPR|Opera', case=False).sum(),
    }
    
    print(f"\nBrowser Distribution:")
    for browser, count in browsers.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  {browser:<15} {count:>8,} ({pct:>5.2f}%)")

def create_statistical_visualizations(df, output_path='statistical_analysis.png'):
    """Create comprehensive statistical visualizations"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. Response time distribution
    ax1 = axes[0, 0]
    ax1.hist(df['response_time'], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Response Time (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Response Time Distribution')
    ax1.axvline(df['response_time'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["response_time"].mean():.0f} ms')
    ax1.legend()
    
    # 2. Status code distribution
    ax2 = axes[0, 1]
    status_counts = df['status'].value_counts().sort_index()
    ax2.bar(status_counts.index.astype(str), status_counts.values, alpha=0.7)
    ax2.set_xlabel('HTTP Status Code')
    ax2.set_ylabel('Count')
    ax2.set_title('HTTP Status Code Distribution')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. HTTP method distribution
    ax3 = axes[1, 0]
    method_counts = df['method'].value_counts()
    ax3.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
    ax3.set_title('HTTP Method Distribution')
    
    # 4. Requests over time
    ax4 = axes[1, 1]
    df_time = df.set_index('timestamp').resample('1min').size()
    ax4.plot(df_time.index, df_time.values, linewidth=1.5)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Requests per Minute')
    ax4.set_title('Request Rate Over Time')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Top IPs
    ax5 = axes[2, 0]
    top_ips = df['ip'].value_counts().head(10)
    ax5.barh(range(len(top_ips)), top_ips.values)
    ax5.set_yticks(range(len(top_ips)))
    ax5.set_yticklabels([ip[:15] + '...' if len(ip) > 15 else ip for ip in top_ips.index])
    ax5.set_xlabel('Request Count')
    ax5.set_title('Top 10 IP Addresses by Request Count')
    ax5.invert_yaxis()
    
    # 6. Response size distribution
    ax6 = axes[2, 1]
    ax6.hist(df['size'], bins=50, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Response Size (bytes)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Response Size Distribution')
    ax6.axvline(df['size'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["size"].mean():.0f} bytes')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n\nStatistical visualizations saved to: {output_path}")

def main():
    """Main execution function"""
    print("Starting Statistical Analysis of Web Server Logs...\n")
    
    # Load data
    print("Loading log file...")
    df = parse_log_file('server.log')
    print(f"Loaded {len(df):,} log entries\n")
    
    # Run all analyses
    compute_basic_statistics(df)
    analyze_http_methods(df)
    analyze_status_codes(df)
    analyze_endpoints(df)
    analyze_ip_addresses(df)
    analyze_temporal_patterns(df)
    analyze_user_agents(df)
    
    # Create visualizations
    create_statistical_visualizations(df)
    
    print("\n" + "="*80)
    print("Statistical analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
