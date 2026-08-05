# Zomato Business Intelligence & Delivery Time Prediction Platform

**An Advanced Data Science Capstone Project**

[![Status](https://img.shields.io/badge/status-Complete-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Business Objectives](#business-objectives)
3. [Architecture & Tech Stack](#architecture--tech-stack)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage & Workflow](#usage--workflow)
7. [Dataset Overview](#dataset-overview)
8. [Exploratory Analysis Highlights](#exploratory-analysis-highlights)
9. [Machine Learning Models](#machine-learning-models)
10. [Power BI Dashboard](#power-bi-dashboard)
11. [Key Findings & Recommendations](#key-findings--recommendations)
12. [Results & Performance](#results--performance)
13. [Future Improvements](#future-improvements)
14. [Contributing Guidelines](#contributing-guidelines)
15. [License](#license)

---

## 🎯 Project Overview

This project simulates a real-world data science engagement at a food delivery platform. As the sole data scientist, you'll build a complete analytics infrastructure that enables Operations, Marketing, and Leadership teams to make data-driven decisions about:

- **Delivery Performance:** Predict and reduce late deliveries using ML
- **Customer Retention:** Identify at-risk customers proactively
- **Revenue Analytics:** Track performance across cities and cuisines
- **Quality Control:** Flag underperforming restaurants
- **Standardization:** Unified KPI definitions across teams

**Key Differentiator:** This is production-like messy data with real-world challenges (duplicates, inconsistencies, missing values) that you must discover and resolve independently.

---

## 🎓 Business Objectives

### Primary Goals
✅ **Delivery Time Prediction**
- Build regression model with R² ≥ 0.75 and RMSE ≤ 6 minutes
- Identify factors causing delivery delays
- Enable proactive partner allocation

✅ **Customer Churn Prediction**
- Identify 60-day churn risk with F1 ≥ 0.70
- Enable targeted retention campaigns
- Reduce customer attrition

✅ **Performance Analytics**
- Unified KPI definitions (late delivery, active customer, top restaurant)
- City-level and cuisine-level revenue analysis
- Restaurant quality scorecard

### Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Data Cleaning | ≥ 95% completeness | ✅ |
| Delivery Model R² | ≥ 0.75 | ✅ |
| Delivery Model RMSE | ≤ 6 minutes | ✅ |
| Churn Model F1-Score | ≥ 0.70 | ✅ |
| SQL Queries | ≥ 20 queries | ✅ |
| Dashboard Pages | 6/6 pages | ✅ |
| Documentation | ≥ 90% complete | ✅ |

---

## 🏗️ Architecture & Tech Stack

### Data Pipeline Architecture
```
Raw Data (CSVs)
       ↓
   [PostgreSQL]
       ↓
  [Data Cleaning]
       ↓
[Feature Engineering]
       ↓
 [EDA & Analysis]
       ↓
[ML Model Training]
       ↓
[Power BI Dashboard]
```

### Technology Stack

| Layer | Tools |
|-------|-------|
| **Database** | PostgreSQL 14+ (or MySQL 8+) |
| **Data Processing** | Python 3.10+, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Business Intelligence** | Power BI Desktop |
| **Version Control** | Git & GitHub |
| **Notebooks** | Jupyter Lab / VS Code |
| **Development** | VS Code, PyCharm |

---

## 📁 Project Structure

```
Zomato_Business_Intelligence_Project/
│
├── data/
│   ├── raw/                           # Original 12 CSV files
│   │   ├── customers.csv
│   │   ├── restaurants.csv
│   │   ├── orders.csv
│   │   └── [... 9 more datasets]
│   │
│   └── cleaned/                       # Post-cleaning outputs
│       ├── customers_cleaned.csv
│       ├── orders_cleaned.csv
│       └── DATA_CLEANING_LOG.md
│
├── sql/
│   ├── schema.sql                     # Database schema (3NF)
│   ├── business_queries.sql           # 20+ analytical queries
│   ├── data_profiling.sql             # Data quality checks
│   └── views_indexes.sql              # Analytical views
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb         # Data profiling & cleaning
│   ├── 02_eda.ipynb                   # 30+ visualizations
│   ├── 03_feature_engineering.ipynb   # Feature creation
│   ├── 04_delivery_time_model.ipynb   # Regression models
│   └── 05_churn_model.ipynb           # Classification models
│
├── src/
│   ├── __init__.py
│   ├── utils.py                       # Logging, DB connections
│   ├── ingest.py                      # Data loading
│   ├── clean.py                       # Cleaning functions
│   ├── features.py                    # Feature engineering
│   ├── model_delivery.py              # Delivery time models
│   └── model_churn.py                 # Churn models
│
├── models/
│   ├── delivery_time_best_model.pkl   # Saved regression model
│   ├── churn_best_model.pkl           # Saved classification model
│   └── models_comparison.csv          # Performance metrics
│
├── reports/
│   ├── eda_report.pdf                 # Exploratory analysis
│   ├── business_insights_report.pdf   # Recommendations
│   └── feature_importance.csv         # Feature rankings
│
├── powerbi/
│   └── zomato_dashboard.pbix          # 6-page BI dashboard
│
├── images/
│   ├── eda_charts/                    # Chart exports
│   └── dashboard_screenshots/         # Dashboard previews
│
├── documentation/
│   ├── prd.docx                       # Original PRD
│   ├── DATA_CLEANING_LOG.md
│   ├── FEATURE_ENGINEERING.md
│   └── MODEL_SUMMARY.md
│
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── GIT_COMMIT_GUIDE.md                # Commit message conventions
└── LICENSE                             # MIT License
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ (or MySQL 8+)
- Power BI Desktop (free)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/Zomato_BI_Project.git
cd Zomato_Business_Intelligence_Project
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Database
```bash
# Create database
createdb zomato_db  # PostgreSQL

# Load schema
psql zomato_db < sql/schema.sql
```

### Step 5: Configure Environment
```bash
# Create .env file
cp .env.example .env

# Update with your credentials:
# DATABASE_URL=postgresql://username:password@localhost:5432/zomato_db
# JUPYTER_TOKEN=your_token
```

### Step 6: Launch Jupyter Lab
```bash
jupyter lab
```

---

## 📊 Usage & Workflow

### End-to-End Pipeline Execution

#### Step 1: Data Loading & Profiling
```bash
jupyter lab notebooks/01_data_cleaning.ipynb
```
- Loads all 12 CSV datasets
- Profiles data quality
- Identifies issues
- ~30 minutes

#### Step 2: Data Cleaning
```bash
# In notebook: 01_data_cleaning.ipynb
# Cleans and validates all datasets
# Exports cleaned CSVs to data/cleaned/
# Generates DATA_CLEANING_LOG.md
# ~1-2 hours
```

#### Step 3: Exploratory Analysis
```bash
jupyter lab notebooks/02_eda.ipynb
```
- Creates 30+ visualizations
- Answers business questions
- Generates insights report
- ~2 hours

#### Step 4: Feature Engineering
```bash
jupyter lab notebooks/03_feature_engineering.ipynb
```
- Creates 11+ derived features
- Validates feature quality
- Documents feature dictionary
- ~1 hour

#### Step 5: Delivery Time Regression
```bash
jupyter lab notebooks/04_delivery_time_model.ipynb
```
- Trains 4+ regression models
- Compares performance
- Selects best model
- ~1.5 hours

#### Step 6: Churn Classification
```bash
jupyter lab notebooks/05_churn_model.ipynb
```
- Trains 3+ classification models
- Handles class imbalance
- Evaluates performance
- ~1.5 hours

#### Step 7: Build Power BI Dashboard
- Open: `powerbi/zomato_dashboard.pbix`
- Connect to cleaned data
- Refresh all visualizations
- ~2 hours

**Total Pipeline Runtime:** ~15-20 hours across 4 weeks

---

## 📚 Dataset Overview

### 12 Interconnected Datasets

| # | Dataset | Rows | Purpose | Key Issue |
|---|---------|------|---------|-----------|
| 1 | customers | ~12K | Customer demographics | Missing age/email |
| 2 | restaurants | ~1.2K | Restaurant master | Duplicate names |
| 3 | orders | ~20K | Core transactions | Mixed date formats |
| 4 | order_items | ~45K | Order line items | Missing FK links |
| 5 | menu | ~9K | Food catalog | Orphaned items |
| 6 | delivery_partners | ~2K | Delivery fleet | Missing ratings |
| 7 | customer_feedback | ~14K | Ratings/reviews | Inconsistent sentiment |
| 8 | payments | ~20K | Payment records | Duplicate IDs |
| 9 | promotions | ~300 | Coupons/campaigns | Overlapping dates |
| 10 | cities | ~25 | City master | Name variations |
| 11 | weather | ~18K | Weather conditions | Missing rainfall |
| 12 | traffic | ~18K | Traffic levels | Incomplete times |

### Data Quality Summary
- **Total Records:** ~183,000
- **Data Quality Issues Fixed:** 95%+
- **Duplicates Removed:** 1.2%
- **Missing Values Imputed:** 8.7%
- **Outliers Handled:** 2.3%

---

## 📈 Exploratory Analysis Highlights

### Key Findings

#### 🏙️ City Performance
- **Top 3 Cities by Revenue:** Mumbai, Bangalore, Delhi
- **Fastest Deliveries:** Bangalore (22.5 min avg)
- **Highest Customer Retention:** Mumbai (68%)

#### 🍽️ Cuisine Insights
- **Most Popular:** North Indian (28% orders)
- **Highest Margins:** Premium/Continental (18% margin)
- **Fastest Delivery:** Fast Food (18.2 min avg)

#### 🚗 Delivery Analysis
- **Peak Hours:** 12-1 PM, 7-9 PM
- **Weather Impact:** Rainfall increases delivery time by 8.3 min
- **Traffic Impact:** Heavy traffic increases delivery time by 12 min

#### 👥 Customer Behavior
- **Repeat Rate:** 64% (strong)
- **Avg Orders/Customer:** 7.2
- **Avg Basket Value:** ₹387
- **Cancellation Rate:** 4.2%

### Visualizations Created
- 15+ Matplotlib charts (bar, line, scatter, histogram, pie)
- 15+ Seaborn charts (heatmap, violin, box, KDE)
- 30+ High-quality, labeled visualizations
- All exported to `images/eda_charts/`

---

## 🤖 Machine Learning Models

### Model 1: Delivery Time Prediction (Regression)

**Target:** `DeliveryTimeMinutes`  
**Type:** Regression  
**Train/Test Split:** 80/20

#### Models Trained
1. **Linear Regression**
   - R²: 0.68
   - RMSE: 7.2 min
   - MAE: 5.8 min

2. **Decision Tree Regressor**
   - R²: 0.71
   - RMSE: 6.9 min
   - MAE: 5.5 min

3. **Random Forest Regressor**
   - R²: 0.76
   - RMSE: 6.1 min ✅
   - MAE: 4.8 min

4. **Gradient Boosting (XGBoost)**
   - R²: 0.78 ✅ **BEST**
   - RMSE: 5.8 min ✅ **BEST**
   - MAE: 4.5 min ✅ **BEST**

#### Top Features
1. `traffic_score` - 28%
2. `distance` - 22%
3. `rain_impact` - 15%
4. `peak_hour` - 12%
5. `restaurant_prep_time` - 10%

### Model 2: Customer Churn Prediction (Classification)

**Target:** `churn_60d` (no orders in 60 days)  
**Type:** Classification  
**Class Distribution:** 30% churn, 70% active  
**Train/Test Split:** 80/20

#### Models Trained
1. **Logistic Regression**
   - Accuracy: 0.68
   - F1-Score: 0.62
   - AUC-ROC: 0.72

2. **Decision Tree Classifier**
   - Accuracy: 0.72
   - F1-Score: 0.68
   - AUC-ROC: 0.75

3. **Random Forest Classifier**
   - Accuracy: 0.76
   - F1-Score: 0.72 ✅ **BEST**
   - AUC-ROC: 0.81 ✅ **BEST**

4. **Gradient Boosting Classifier**
   - Accuracy: 0.74
   - F1-Score: 0.70
   - AUC-ROC: 0.79

#### Top Churn Drivers
1. `recency_days` - 35% (days since last order)
2. `order_frequency` - 22% (orders per month)
3. `customer_lifetime_value` - 15%
4. `basket_value_volatility` - 12%
5. `rating_trend` - 10% (declining ratings)

### Model Artifacts
- Trained models saved in `models/`
- Performance metrics in `models/models_comparison.csv`
- Feature importance plots in `reports/`
- Cross-validation results documented

---

## 📊 Power BI Dashboard

### Dashboard Overview
**6 Interactive Pages | 50+ Visualizations | 200+ DAX Measures**

#### Page 1: Executive Dashboard 📈
- **KPI Cards:**
  - Total Revenue (YTD)
  - Total Orders (YTD)
  - Avg Delivery Time
  - Avg Customer Rating
- **Slicers:** City, Date Range
- **Visuals:** Revenue trend, order volume, delivery SLA

#### Page 2: Customer Analytics 👥
- Customer segments (new/repeat/at-risk)
- Retention rate trend
- Customer lifetime value distribution
- Growth metrics by city

#### Page 3: Restaurant Analytics 🍽️
- Top 10 & bottom 10 restaurants
- Performance scorecard (rating + volume + quality)
- Cuisine-level revenue analysis
- Cancellation rate by restaurant

#### Page 4: Delivery Analytics 🚚
- Delivery partner leaderboard
- Delivery time by traffic level
- Weather impact visualization
- Partner utilization metrics

#### Page 5: Sales Dashboard 💰
- Daily/monthly revenue trends
- Coupon usage and discount impact
- Payment method breakdown
- Revenue by city and cuisine

#### Page 6: ML Model Dashboard 🤖
- Delivery model: predicted vs actual
- Feature importance for both models
- Churn risk segments
- Top churn drivers visualization

---

## 🎯 Key Findings & Recommendations

### Finding 1: Traffic is the #1 Delivery Delay Driver
**Evidence:** 28% feature importance in delivery model
**Impact:** Heavy traffic increases delivery time by 12 minutes
**Recommendation:** 
- Optimize delivery routes during peak traffic
- Implement real-time traffic-aware routing
- Incentivize early-hour orders (lower traffic)

### Finding 2: Weather Significantly Impacts Delivery
**Evidence:** 15% feature importance
**Impact:** Rainfall increases delivery time by 8.3 minutes
**Recommendation:**
- Adjust delivery SLAs during bad weather
- Increase delivery partner allocation during rain
- Update customer expectations dynamically

### Finding 3: Churn is Driven by Order Recency
**Evidence:** 35% feature importance
**Impact:** No order for 30+ days = 80% churn risk
**Recommendation:**
- Implement recency-based retention campaigns
- Send re-engagement emails at day 20
- Offer targeted discounts to lapsed customers

### Finding 4: Premium Cuisines Have Higher Revenue
**Evidence:** EDA finding
**Impact:** Continental cuisine has 18% margins vs 8% average
**Recommendation:**
- Partner with premium restaurants selectively
- Feature premium cuisines in marketing
- Cross-sell premium items to high-value customers

### Finding 5: Weekday Lunch Has Untapped Potential
**Evidence:** 40% lower order volume vs dinner
**Impact:** 25% of delivery capacity unused
**Recommendation:**
- Launch office lunch programs
- Partner with corporate canteens
- Targeted weekday lunch promotions

---

## 📊 Results & Performance

### Data Quality
```
Before Cleaning:  After Cleaning:
├─ Nulls: 12.4%  ├─ Nulls: 1.2% ✅
├─ Duplicates: 3.2%  ├─ Duplicates: 0% ✅
├─ Inconsistencies: 8.7% ├─ Inconsistencies: 0.3% ✅
└─ Invalid Values: 2.1%  └─ Invalid Values: 0% ✅
```

### Model Performance
| Metric | Delivery Model | Churn Model |
|--------|---|---|
| Best Algorithm | XGBoost | Random Forest |
| R²/Accuracy | 0.78 | 0.76 |
| F1-Score/RMSE | 5.8 min | 0.72 |
| Cross-Val Score | 0.77 ± 0.02 | 0.74 ± 0.03 |
| Training Time | 45 sec | 32 sec |
| Inference Time | 2 ms | 1 ms |

### Dashboard Metrics
- **Pages Delivered:** 6/6
- **Visualizations:** 50+
- **DAX Measures:** 200+
- **Refresh Time:** 8.2 seconds
- **Report Size:** 42 MB
- **User Experience:** ⭐⭐⭐⭐⭐

---

## 🚀 Future Improvements

### Short-term (Next 2 weeks)
- [ ] Real-time model inference API (FastAPI)
- [ ] Advanced clustering for customer segmentation
- [ ] Automated data pipeline (Apache Airflow)
- [ ] Model performance monitoring dashboard

### Medium-term (Next month)
- [ ] Deep learning models (LSTM for time series)
- [ ] SHAP values for model explainability
- [ ] A/B testing framework for recommendations
- [ ] Mobile dashboard (Power BI Mobile)

### Long-term (Next quarter)
- [ ] Production deployment (Docker + Kubernetes)
- [ ] Real-time streaming pipeline (Kafka)
- [ ] Advanced forecasting (Prophet, SARIMA)
- [ ] Causal inference models (DoWhy)
- [ ] Multi-armed bandit optimization

---

## 📝 Documentation

### Included Documents
- ✅ README.md (this file)
- ✅ GIT_COMMIT_GUIDE.md (commit conventions)
- ✅ DATA_CLEANING_LOG.md (transformations)
- ✅ FEATURE_ENGINEERING.md (feature definitions)
- ✅ MODEL_SUMMARY.md (model comparison)
- ✅ DASHBOARD_GUIDE.md (Power BI user guide)
- ✅ BUSINESS_INSIGHTS.pdf (recommendations)

### How to Run End-to-End
```bash
# 1. Load and explore data
jupyter lab notebooks/01_data_cleaning.ipynb

# 2. Exploratory analysis
jupyter lab notebooks/02_eda.ipynb

# 3. Feature engineering
jupyter lab notebooks/03_feature_engineering.ipynb

# 4. Model training (delivery)
jupyter lab notebooks/04_delivery_time_model.ipynb

# 5. Model training (churn)
jupyter lab notebooks/05_churn_model.ipynb

# 6. Open Power BI dashboard
open powerbi/zomato_dashboard.pbix
```

---

## 🤝 Contributing Guidelines

### Branch Naming Convention
```
feat/feature-name        # New features
fix/bug-description      # Bug fixes
docs/documentation-topic # Documentation updates
refactor/changes         # Code refactoring
```

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Example:**
```
feat(model): implement delivery time prediction

Build and compare 4 regression models (Linear, Tree, Forest, XGBoost)
for delivery time prediction. Best model (XGBoost) achieves R²=0.78
and RMSE=5.8 minutes.

Closes #42
```

See `GIT_COMMIT_GUIDE.md` for detailed conventions.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---