# Task 3: DDoS Detection Using Regression Analysis

**Author:** Tornike Tchabukiani (t_tchabukiani25_16928)  
**Date:** February 12, 2026

---

## 📄 Main Report

**[ddos.md](./ddos.md)** - Complete comprehensive report (1,286 lines, 41KB)

This single document contains:
- **Attack Detection Results** - Time intervals and severity classification
- **Complete Analysis Report** - 9 main sections with methodology, results, and conclusions
- **Appendix A** - Quick start guide and file listing
- **Appendix B** - Complete program code documentation

---

## 🚨 Attack Summary

**Multi-Wave DDoS Attack Detected:**

| Wave | Time | Duration | Peak Traffic | Z-Score | Severity |
|------|------|----------|--------------|---------|----------|
| #1 | 18:40-18:41 | 2 min | 6,786 req/min | 2.22 | Moderate |
| #2 | 18:43-18:44 | 2 min | 12,292 req/min | 4.74 | **CRITICAL** |

**Pattern:** Coordinated two-wave botnet attack with 1-minute strategic pause

---

## 📁 Repository Contents

### 📄 Documentation
- **ddos.md** (41KB) - Complete merged report with all documentation

### 💻 Python Programs (5 modules)
- **ddos_analysis.py** - Complete integrated analysis pipeline
- **statistical_analysis.py** - Statistical data extraction
- **regression_analysis.py** - Regression modeling
- **attack_detection.py** - Attack classification
- **enhanced_visualization.py** - Publication-quality 300 DPI visualization

### 📊 Visualizations
- **enhanced_ddos_analysis.png** ⭐ - 8-panel dashboard (1.6MB, 300 DPI)
- **ddos_analysis.png** - 4-panel analysis (298KB, 300 DPI)
- **statistical_analysis.png** - 6-panel statistics (249KB)
- **regression_analysis.png** - 6-panel diagnostics (272KB)

### 📝 Data Files
- **t_tchabukiani25_16928_server.log** - Original log (84,665 entries)
- **attack_summary.txt** - Brief attack summary

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Run Complete Analysis
```bash
python ddos_analysis.py
```

### Generate Enhanced Visualization
```bash
python enhanced_visualization.py
```

### Read Full Report
Open **[ddos.md](./ddos.md)** for complete documentation

---

## 📊 Key Results

- **Dataset:** 84,665 requests over 61 minutes
- **Baseline:** 1,388 requests/minute
- **Attack Peak:** 12,292 requests/minute (8.9x amplification)
- **Statistical Significance:** Z-score 4.74 (p < 0.0001)
- **Server Impact:** 57% error rate during peak attack
- **Severity:** CRITICAL (9/11 score)

---

## 📖 Documentation Structure

The **ddos.md** file is organized as follows:

1. **Attack Time Intervals** - Immediate results
2. **Executive Summary** - Key findings
3. **Main Report** (Sections 1-9):
   - Introduction
   - Methodology
   - Data Analysis  
   - Regression Analysis
   - Results
   - Visualizations
   - Code Implementation
   - Conclusions
   - References
4. **Appendix A** - Quick Start Guide
5. **Appendix B** - Program Documentation

---

## 🎯 For Reviewers

**Start here:** [ddos.md](./ddos.md) - Section: "🚨 DETECTED DDoS ATTACK TIME INTERVALS"

**For technical details:** See Section 2 (Methodology) and Section 4 (Regression Analysis)

**For code details:** See Appendix B (Program Code Documentation)

**For reproduction:** See Appendix A (Quick Start Guide)

---

**All documentation consolidated into a single comprehensive ddos.md file**
