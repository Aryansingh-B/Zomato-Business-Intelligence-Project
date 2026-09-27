#!/usr/bin/env python3
"""
generate_sample_data.py
=======================
Creates sample CSV files for testing the Zomato BI project.

Run this script to generate test data in data/raw/
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("🔧 Generating Sample CSV Data for Zomato Project...\n")

# Create data/raw folder
os.makedirs('data/raw', exist_ok=True)

# ============================================================================
# 1. Cities (25 rows)
# ============================================================================
cities = pd.DataFrame({
    'city': ['Mumbai', 'Bangalore', 'Delhi', 'Pune', 'Chennai'],
})
cities = pd.concat([cities] * 5, ignore_index=True)
cities['cityid'] = range(1, len(cities) + 1)
cities = cities[['cityid', 'city']]
cities.to_csv('data/raw/cities.csv', index=False)
print(f"✅ cities.csv - {len(cities)} rows")

# ============================================================================
# 2. Customers (100 rows)
# ============================================================================
customers = pd.DataFrame({
    'customerid': range(1, 101),
    'name': [f'Customer_{i}' for i in range(1, 101)],
    'age': np.random.randint(18, 70, 100),
    'email': [f'customer{i}@email.com' for i in range(1, 101)],
    'phone': [f'9{np.random.randint(1000000000, 9999999999)}' for _ in range(100)],
    'city': np.random.choice(['Mumbai', 'Bangalore', 'Delhi', 'Pune', 'Chennai'], 100),
    'registrationdate': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(100)],
})
customers.to_csv('data/raw/customers.csv', index=False)
print(f"✅ customers.csv - {len(customers)} rows")

# ============================================================================
# 3. Restaurants (50 rows)
# ============================================================================
restaurants = pd.DataFrame({
    'restaurantid': range(1, 51),
    'restaurantname': [f'Restaurant_{i}' for i in range(1, 51)],
    'cuisine': np.random.choice(['North Indian', 'South Indian', 'Chinese', 'Italian', 'Continental'], 50),
    'city': np.random.choice(['Mumbai', 'Bangalore', 'Delhi', 'Pune', 'Chennai'], 50),
    'rating': np.random.uniform(3.0, 5.0, 50),
})
restaurants.to_csv('data/raw/restaurants.csv', index=False)
print(f"✅ restaurants.csv - {len(restaurants)} rows")

# ============================================================================
# 4. Menu (200 rows)
# ============================================================================
menu = pd.DataFrame({
    'fooditemid': range(1, 201),
    'restaurantid': np.random.randint(1, 51, 200),
    'foodname': [f'Dish_{i}' for i in range(1, 201)],
    'price': np.random.uniform(50, 500, 200),
    'availability': np.random.choice(['Yes', 'No'], 200),
})
menu.to_csv('data/raw/menu.csv', index=False)
print(f"✅ menu.csv - {len(menu)} rows")

# ============================================================================
# 5. Delivery Partners (30 rows)
# ============================================================================
delivery_partners = pd.DataFrame({
    'deliverypartnerid': range(1, 31),
    'name': [f'Partner_{i}' for i in range(1, 31)],
    'age': np.random.randint(20, 50, 30),
    'rating': np.random.uniform(3.0, 5.0, 30),
})
delivery_partners.to_csv('data/raw/delivery_partners.csv', index=False)
print(f"✅ delivery_partners.csv - {len(delivery_partners)} rows")

# ============================================================================
# 6. Orders (500 rows) - MAIN TABLE
# ============================================================================
orders = pd.DataFrame({
    'orderid': range(1, 501),
    'customerid': np.random.randint(1, 101, 500),
    'restaurantid': np.random.randint(1, 51, 500),
    'deliverypartnerid': np.random.randint(1, 31, 500),
    'orderdate': [datetime.now() - timedelta(days=np.random.randint(0, 90)) for _ in range(500)],
    'ordertime': [f'{np.random.randint(8, 23)}:{np.random.randint(0, 60):02d}:00' for _ in range(500)],
    'deliverytimeminutes': np.random.randint(15, 60, 500),
    'foodcost': np.random.uniform(100, 1000, 500),
    'deliveryfee': np.random.uniform(10, 100, 500),
    'discount': np.random.uniform(0, 200, 500),
    'finalamount': np.random.uniform(200, 1200, 500),
    'paymentmethod': np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Cash'], 500),
})
orders.to_csv('data/raw/orders.csv', index=False)
print(f"✅ orders.csv - {len(orders)} rows")

# ============================================================================
# 7. Order Items (1500 rows)
# ============================================================================
order_items = pd.DataFrame({
    'orderitemid': range(1, 1501),
    'orderid': np.random.randint(1, 501, 1500),
    'fooditemid': np.random.randint(1, 201, 1500),
    'quantity': np.random.randint(1, 5, 1500),
})
order_items.to_csv('data/raw/order_items.csv', index=False)
print(f"✅ order_items.csv - {len(order_items)} rows")

# ============================================================================
# 8. Payments (500 rows)
# ============================================================================
payments = pd.DataFrame({
    'paymentid': range(1, 501),
    'orderid': range(1, 501),
    'paymentmethod': np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Cash'], 500),
    'paymentstatus': np.random.choice(['Success', 'Failed', 'Pending'], 500),
    'paymentdate': [datetime.now() - timedelta(days=np.random.randint(0, 90)) for _ in range(500)],
})
payments.to_csv('data/raw/payments.csv', index=False)
print(f"✅ payments.csv - {len(payments)} rows")

# ============================================================================
# 9. Customer Feedback (500 rows)
# ============================================================================
customer_feedback = pd.DataFrame({
    'feedbackid': range(1, 501),
    'orderid': range(1, 501),
    'customerrating': np.random.randint(1, 6, 500),
    'deliveryrating': np.random.randint(1, 6, 500),
    'foodrating': np.random.randint(1, 6, 500),
    'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], 500),
})
customer_feedback.to_csv('data/raw/customer_feedback.csv', index=False)
print(f"✅ customer_feedback.csv - {len(customer_feedback)} rows")

# ============================================================================
# 10. Promotions (50 rows)
# ============================================================================
promotions = pd.DataFrame({
    'promotionid': range(1, 51),
    'couponcode': [f'COUPON_{i}' for i in range(1, 51)],
    'discountpercentage': np.random.uniform(5, 50, 50),
    'expirydate': [datetime.now() + timedelta(days=np.random.randint(1, 365)) for _ in range(50)],
})
promotions.to_csv('data/raw/promotions.csv', index=False)
print(f"✅ promotions.csv - {len(promotions)} rows")

# ============================================================================
# 11. Weather (500 rows)
# ============================================================================
weather = pd.DataFrame({
    'weatherid': range(1, 501),
    'city': np.random.choice(['Mumbai', 'Bangalore', 'Delhi', 'Pune', 'Chennai'], 500),
    'date': [datetime.now() - timedelta(days=np.random.randint(0, 90)) for _ in range(500)],
    'weathercondition': np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Heavy Rain', 'Foggy'], 500),
    'rainfall': np.random.uniform(0, 20, 500),
})
weather.to_csv('data/raw/weather.csv', index=False)
print(f"✅ weather.csv - {len(weather)} rows")

# ============================================================================
# 12. Traffic (500 rows)
# ============================================================================
traffic = pd.DataFrame({
    'trafficid': range(1, 501),
    'city': np.random.choice(['Mumbai', 'Bangalore', 'Delhi', 'Pune', 'Chennai'], 500),
    'date': [datetime.now() - timedelta(days=np.random.randint(0, 90)) for _ in range(500)],
    'trafficlevel': np.random.choice(['Low', 'Medium', 'High', 'Very High'], 500),
})
traffic.to_csv('data/raw/traffic.csv', index=False)
print(f"✅ traffic.csv - {len(traffic)} rows")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ ALL SAMPLE DATA GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\n📁 Location: data/raw/")
print(f"📊 Total files: 12")
print(f"📈 Total rows: ~5,500 rows")
print(f"\nNow you can run the notebooks! 🚀")
print(f"\nNext steps:")
print(f"  1. Go to notebooks/")
print(f"  2. Open 01_data_cleaning.ipynb")
print(f"  3. Click 'Run All'")
print(f"  4. Watch the magic happen! ✨")
