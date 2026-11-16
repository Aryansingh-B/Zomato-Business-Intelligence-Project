# Zomato Business Intelligence & Delivery Time Prediction Platform

> **Turning Food Delivery Data into Actionable Business Decisions**
>
> An end-to-end analytics project built using **Python, PostgreSQL, Machine Learning, and Power BI** to analyze delivery performance, customer churn, restaurant quality, and revenue trends. The project processes **183K+ records across 12 datasets**, with an XGBoost delivery prediction model achieving **R² = 0.78** and a Random Forest churn model achieving **F1-score = 0.72**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791)
![Power BI](https://img.shields.io/badge/BI-Power%20BI-yellow)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**Quick Links:** [Dashboard](#dashboard-preview) · [Dataset](#dataset-overview) · [Key Findings](#key-insights--findings)

---

## 1. Business Problem

Food delivery platforms face operational challenges such as delayed deliveries, customer churn, inconsistent restaurant performance, and revenue fluctuations.

These issues can increase operating costs, reduce customer satisfaction, and affect long-term revenue.

**Business goal:** Use data analytics and predictive modeling to identify performance gaps, understand customer behavior, and support better operational and retention decisions.

## 2. Dataset Overview

The project uses **12 interconnected CSV datasets** containing approximately **183,000 records** across customer, restaurant, order, payment, delivery, and operational data.

**Source:** Project dataset containing simulated food-delivery business data. The source file does not specify a public dataset URL or a defined time period.

| Dataset | Approx. Records | Business Purpose |
|---|---:|---|
| Customers | 12K | Customer demographics |
| Restaurants | 1.2K | Restaurant performance |
| Orders | 20K | Order and delivery analysis |
| Order Items | 45K | Item-level transactions |
| Menu | 9K | Food catalog analysis |
| Delivery Partners | 2K | Delivery fleet performance |
| Customer Feedback | 14K | Ratings and reviews |
| Payments | 20K | Payment analysis |
| Promotions | 300 | Campaign performance |
| Cities | 25 | City-level analysis |
| Weather | 18K | Weather impact |
| Traffic | 18K | Traffic conditions |

**Key fields:** Customer ID, Order ID, Restaurant ID, City, Cuisine, Delivery Time, Order Date, Payment Method, Rating, Traffic Score, and Weather Conditions.

## 3. Project Objectives & Business Questions

### Objectives
- Analyze revenue, orders, and restaurant performance across cities and cuisines.
- Identify factors contributing to delivery delays.
- Understand customer behavior and predict 60-day churn risk.
- Build interactive dashboards for business performance monitoring.

### Business Questions
1. Which cities generate the highest revenue?
2. Which cuisines have the strongest order volume and margins?
3. How do traffic and rainfall affect delivery time?
4. Which factors are most strongly associated with customer churn?
5. Which restaurants have the strongest and weakest performance?
6. When does order demand peak, and where is delivery capacity underutilized?

## 4. Methodology & Analytical Workflow

| Stage | Work Performed | Tools |
|---|---|---|
| Data Cleaning | Handled missing values, duplicates, inconsistent formats, and outliers | Python, Pandas |
| Database Analysis | Designed a 3NF schema, created analytical views, and wrote 20+ business queries | PostgreSQL |
| EDA | Created 30+ visualizations to identify trends and performance gaps | Matplotlib, Seaborn |
| Feature Engineering | Developed 11+ derived features for analysis and modeling | Python |
| Predictive Modeling | Compared regression and classification models | Scikit-learn, XGBoost |
| Business Intelligence | Built a 6-page dashboard with 50+ visualizations and 200+ DAX measures | Power BI |

### Model Performance

| Use Case | Best Model | Key Result |
|---|---|---|
| Delivery Time Prediction | XGBoost | R²: **0.78**, RMSE: **5.8 min** |
| Customer Churn Prediction | Random Forest | F1-score: **0.72**, ROC-AUC: **0.81** |

## 5. Dashboard Preview

The Power BI dashboard contains six interactive pages covering business performance, customer behavior, restaurants, delivery operations, sales, and model insights.

### Executive Dashboard
![Executive Dashboard](images/dashboard_screenshots/executive_dashboard.png)

*Revenue, order volume, average delivery time, customer ratings, and delivery trends.*

### Customer Analytics
![Customer Analytics](images/dashboard_screenshots/customer_analytics.png)

*Customer segments, retention trends, and customer lifetime value.*

### Delivery Analytics
![Delivery Analytics](images/dashboard_screenshots/delivery_analytics.png)

*Delivery performance across traffic conditions, weather, and delivery partners.*

> Replace the image paths above with the actual screenshot filenames in your repository.

## 6. Key Insights & Findings

- **Delivery delays:** Heavy traffic increases average delivery time by **12 minutes**, while rainfall adds **8.3 minutes**.
- **Delivery prediction:** Traffic score contributes **28%** of the best model's feature importance; distance contributes **22%**.
- **Customer churn:** Order recency is the leading churn driver, contributing **35%** of feature importance.
- **Customer retention:** Customers with no order for 30+ days show an estimated **80% churn risk**.
- **City performance:** Mumbai, Bangalore, and Delhi are the top three cities by revenue. Bangalore has the fastest average delivery time at **22.5 minutes**.
- **Cuisine performance:** North Indian cuisine accounts for **28% of orders**, while Continental cuisine shows an **18% margin** compared with the reported 8% average.

## 7. Actionable Business Recommendations

| Finding | Recommendation | Expected Business Value |
|---|---|---|
| Traffic adds 12 minutes to delivery time | Use traffic-aware route planning and peak-hour partner allocation | Reduce delivery delays |
| Rainfall adds 8.3 minutes | Adjust delivery estimates and partner capacity during bad weather | Improve delivery reliability |
| Recency is the top churn driver | Trigger targeted re-engagement campaigns around day 20 without an order | Support customer retention |
| Continental cuisine has an 18% margin | Explore partnerships and promotions with high-margin restaurants | Support margin growth |
| Weekday lunch orders are 40% lower than dinner | Test office lunch offers and corporate partnerships | Improve off-peak capacity utilization |

*Business impact should be validated through controlled campaigns and operational testing; these recommendations are opportunities, not measured financial savings.*

## 8. Repository Structure & How to Run

```text
Zomato_Business_Intelligence_Project/
│
├── data/
│   ├── raw/                 # 12 original CSV datasets
│   └── cleaned/             # Cleaned datasets
│
├── sql/
│   ├── schema.sql
│   ├── business_queries.sql
│   ├── data_profiling.sql
│   └── views_indexes.sql
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_delivery_time_model.ipynb
│   └── 05_churn_model.ipynb
│
├── src/                     # Reusable Python scripts
├── models/                  # Trained model artifacts
├── reports/                 # Analysis and business reports
├── powerbi/
│   └── zomato_dashboard.pbix
├── images/
│   ├── eda_charts/
│   └── dashboard_screenshots/
│
├── documentation/
├── requirements.txt
└── README.md
