# ============================================================
# CUSTOMER SEGMENTATION AND PURCHASE PREDICTION PROJECT
# Dataset : OnlineRetail.csv
# Author : Pratap Reddy
# ============================================================

# ============================================================
# STEP 1 : IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")

plt.style.use("ggplot")

# ============================================================
# STEP 2 : OPTIONAL SQL DATA EXTRACTION
# ============================================================

"""
If your data is stored in SQL Server instead of CSV,
you can use the following code.

import pyodbc

conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=YOUR_SERVER;"
    "Database=YOUR_DATABASE;"
    "Trusted_Connection=yes;"
)

query = "SELECT * FROM OnlineRetail"

df = pd.read_sql(query, conn)

"""

# ============================================================
# STEP 3 : LOAD DATASET
# ============================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(
    "OnlineRetail.csv",
    encoding="ISO-8859-1"
)

print("\nDataset Loaded Successfully")

# ============================================================
# STEP 4 : DATA UNDERSTANDING
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("\nShape of Dataset")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nFirst Five Records")
print(df.head())

print("\nLast Five Records")
print(df.tail())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

# ============================================================
# STEP 5 : DATA CLEANING
# ============================================================

print("\nCleaning Dataset...")

# Remove rows where CustomerID is missing
df = df.dropna(subset=['CustomerID'])

# Remove duplicate records
df = df.drop_duplicates()

# Remove negative quantities
df = df[df['Quantity'] > 0]

# Remove zero or negative prices
df = df[df['UnitPrice'] > 0]

# Convert CustomerID to integer
df['CustomerID'] = df['CustomerID'].astype(int)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("Cleaning Completed Successfully")

print("\nRemaining Records :", len(df))

# ============================================================
# STEP 6 : OUTLIER REMOVAL (IQR METHOD)
# ============================================================

print("\nRemoving Extreme Outliers...")

Q1 = df['Quantity'].quantile(0.25)
Q3 = df['Quantity'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df['Quantity'] >= lower) &
    (df['Quantity'] <= upper)
]

print("Outlier Removal Completed")

# ============================================================
# STEP 7 : FEATURE ENGINEERING
# ============================================================

print("\nCreating New Features...")

# ------------------------------------------------------------
# Total Revenue
# ------------------------------------------------------------

df['TotalAmount'] = (
    df['Quantity'] *
    df['UnitPrice']
)

# ------------------------------------------------------------
# Invoice Month
# ------------------------------------------------------------

df['Month'] = (
    df['InvoiceDate']
    .dt.to_period('M')
)

# ------------------------------------------------------------
# Invoice Year
# ------------------------------------------------------------

df['Year'] = (
    df['InvoiceDate']
    .dt.year
)

# ------------------------------------------------------------
# Invoice Day
# ------------------------------------------------------------

df['Day'] = (
    df['InvoiceDate']
    .dt.day_name()
)

# ============================================================
# STEP 8 : CUSTOMER LEVEL FEATURES
# ============================================================

print("\nCreating Customer Features...")

customer_features = (

    df.groupby("CustomerID")

    .agg(

        TotalRevenue=('TotalAmount', 'sum'),

        TotalOrders=('InvoiceNo', 'nunique'),

        TotalQuantity=('Quantity', 'sum'),

        AverageOrderValue=('TotalAmount', 'mean')

    )

)

# ------------------------------------------------------------
# Customer Lifetime Value (Simple Estimation)
# ------------------------------------------------------------

customer_features['CustomerLifetimeValue'] = (

    customer_features['TotalRevenue']

)

# ------------------------------------------------------------
# Revenue Per Order
# ------------------------------------------------------------

customer_features['RevenuePerOrder'] = (

    customer_features['TotalRevenue']

    /

    customer_features['TotalOrders']

)

# ------------------------------------------------------------
# Purchase Frequency
# ------------------------------------------------------------

customer_features['PurchaseFrequency'] = (

    customer_features['TotalOrders']

)

print("\nCustomer Features Created Successfully")

print(customer_features.head())

# ============================================================
# STEP 9 : KPI SUMMARY
# ============================================================

print("\n========== KPI SUMMARY ==========")

print("Total Customers :", df['CustomerID'].nunique())

print("Total Orders :", df['InvoiceNo'].nunique())

print("Total Products :", df['StockCode'].nunique())

print("Total Countries :", df['Country'].nunique())

print("Total Revenue : £{:,.2f}".format(df['TotalAmount'].sum()))

print("Average Order Value : £{:,.2f}".format(
    customer_features['AverageOrderValue'].mean()
))

print("Average Revenue Per Customer : £{:,.2f}".format(
    customer_features['TotalRevenue'].mean()
))

print("=" * 60)
print("PART 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# PART 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "="*60)
print("PART 2 : EXPLORATORY DATA ANALYSIS")
print("="*60)

sns.set_style("whitegrid")

# ------------------------------------------------------------
# 1. Dataset Summary
# ------------------------------------------------------------

print("\nSummary Statistics")
print(df.describe())

print("\nData Types")
print(df.dtypes)

# ------------------------------------------------------------
# 2. Sales Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(df["TotalAmount"], bins=50, kde=True)

plt.title("Sales Distribution")
plt.xlabel("Total Amount")
plt.ylabel("Frequency")

plt.show()

# ------------------------------------------------------------
# 3. Quantity Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(df["Quantity"], bins=40, color="green")

plt.title("Quantity Distribution")

plt.xlabel("Quantity")

plt.ylabel("Frequency")

plt.show()

# ------------------------------------------------------------
# 4. Unit Price Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(df["UnitPrice"], bins=40, color="orange")

plt.title("Unit Price Distribution")

plt.xlabel("Unit Price")

plt.ylabel("Frequency")

plt.show()

# ------------------------------------------------------------
# 5. Top 10 Countries by Sales
# ------------------------------------------------------------

country_sales = (
    df.groupby("Country")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

country_sales.plot(kind="bar")

plt.title("Top 10 Countries by Sales")

plt.xlabel("Country")

plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.show()

# ------------------------------------------------------------
# 6. Monthly Sales Trend
# ------------------------------------------------------------

monthly_sales = (
    df.groupby("Month")["TotalAmount"]
    .sum()
)

plt.figure(figsize=(10,5))

monthly_sales.plot(marker="o")

plt.title("Monthly Sales Trend")

plt.xlabel("Month")

plt.ylabel("Sales")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# 7. Top 10 Products
# ------------------------------------------------------------

top_products = (
    df.groupby("Description")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

top_products.plot(kind="barh")

plt.title("Top 10 Products by Sales")

plt.xlabel("Sales")

plt.ylabel("Product")

plt.show()

# ------------------------------------------------------------
# 8. Top Customers
# ------------------------------------------------------------

top_customers = (
    df.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

top_customers.plot(kind="bar")

plt.title("Top 10 Customers")

plt.xlabel("Customer ID")

plt.ylabel("Sales")

plt.show()

# ------------------------------------------------------------
# 9. Sales by Weekday
# ------------------------------------------------------------

weekday_sales = (
    df.groupby("Day")["TotalAmount"].sum())

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_sales = weekday_sales.reindex(weekday_order)

plt.figure(figsize=(10,5))

weekday_sales.plot(kind="bar")

plt.title("Sales by Weekday")

plt.xlabel("Weekday")

plt.ylabel("Sales")

plt.show()

# ------------------------------------------------------------
# 10. Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

corr = df[
    ["Quantity",
     "UnitPrice",
     "TotalAmount"]
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# ------------------------------------------------------------
# 11. Top 10 Customers by Sales (Bar Chart)
# ------------------------------------------------------------

customer_sales = (
    df.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

customer_sales.plot(kind="bar", color="skyblue")

plt.title("Top 10 Customers by Sales")
plt.xlabel("Customer ID")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# 12. Scatter Plot
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    df["Quantity"],
    df["TotalAmount"],
    alpha=0.4
)

plt.title("Quantity vs Sales")

plt.xlabel("Quantity")

plt.ylabel("Total Amount")

plt.show()

# ------------------------------------------------------------
# 13. Pair Plot
# ------------------------------------------------------------

sns.pairplot(
    df[
        ["Quantity",
         "UnitPrice",
         "TotalAmount"]
    ]
)

plt.show()

# ------------------------------------------------------------
# 14. Sales by Year
# ------------------------------------------------------------

year_sales = (
    df.groupby("Year")["TotalAmount"]
    .sum()
)

plt.figure(figsize=(8,5))

year_sales.plot(kind="bar")

plt.title("Yearly Sales")

plt.xlabel("Year")

plt.ylabel("Sales")

plt.show()

# ------------------------------------------------------------
# 15. Save Clean Dataset
# ------------------------------------------------------------

df.to_csv(
    "clean_customer_data.csv",
    index=False
)

print("\nClean dataset saved successfully.")

print("="*60)
print("PART 2 COMPLETED SUCCESSFULLY")
print("="*60)

# ============================================================
# PART 3: RFM ANALYSIS
# ============================================================

print("\n" + "="*60)
print("PART 3 : RFM ANALYSIS")
print("="*60)

# ------------------------------------------------------------
# Create Snapshot Date
# ------------------------------------------------------------

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# ------------------------------------------------------------
# Calculate RFM
# ------------------------------------------------------------

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalAmount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print("\nFirst Five RFM Records")
print(rfm.head())

# ------------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------------

print("\nRFM Summary")
print(rfm.describe())

# ------------------------------------------------------------
# Assign RFM Scores
# ------------------------------------------------------------

rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5,4,3,2,1]
)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1,2,3,4,5]
)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"],
    5,
    labels=[1,2,3,4,5]
)

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str) +
    rfm["F_Score"].astype(str) +
    rfm["M_Score"].astype(str)
)

print("\nRFM Scores")
print(rfm.head())

# ------------------------------------------------------------
# Recency Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.hist(rfm["Recency"], bins=30)

plt.title("Recency Distribution")
plt.xlabel("Days Since Last Purchase")
plt.ylabel("Number of Customers")

plt.show()

# ------------------------------------------------------------
# Frequency Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.hist(rfm["Frequency"], bins=25)

plt.title("Frequency Distribution")
plt.xlabel("Number of Purchases")
plt.ylabel("Customers")

plt.show()

# ------------------------------------------------------------
# Monetary Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.hist(rfm["Monetary"], bins=30)

plt.title("Monetary Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Customers")

plt.show()

# ------------------------------------------------------------
# Top 10 Customers by Monetary Value
# ------------------------------------------------------------

top_monetary = (
    rfm.sort_values("Monetary", ascending=False)
       .head(10)
)

plt.figure(figsize=(12,6))

plt.bar(
    top_monetary.index.astype(str),
    top_monetary["Monetary"]
)

plt.title("Top 10 Customers by Monetary Value")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Top 10 Customers by Frequency
# ------------------------------------------------------------

top_frequency = (
    rfm.sort_values("Frequency", ascending=False)
       .head(10)
)

plt.figure(figsize=(12,6))

plt.bar(
    top_frequency.index.astype(str),
    top_frequency["Frequency"]
)

plt.title("Top 10 Customers by Purchase Frequency")
plt.xlabel("Customer ID")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# RFM Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

sns.heatmap(
    rfm[["Recency","Frequency","Monetary"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("RFM Correlation Heatmap")

plt.show()

# ------------------------------------------------------------
# Save RFM Dataset
# ------------------------------------------------------------

rfm.to_csv("rfm_customer_data.csv")

print("\nRFM Dataset Saved Successfully.")

print("="*60)
print("PART 3 COMPLETED SUCCESSFULLY")
print("="*60)

# ============================================================
# PART 4: CUSTOMER SEGMENTATION USING K-MEANS
# ============================================================

print("\n" + "="*60)
print("PART 4 : CUSTOMER SEGMENTATION")
print("="*60)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# Standardize RFM Data
# ------------------------------------------------------------

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm[["Recency", "Frequency", "Monetary"]]
)

# ------------------------------------------------------------
# Elbow Method
# ------------------------------------------------------------

wcss = []

for i in range(2, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(rfm_scaled)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(2,11), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# Final K-Means Model
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# ------------------------------------------------------------
# Silhouette Score
# ------------------------------------------------------------

score = silhouette_score(rfm_scaled, rfm["Cluster"])

print("\nSilhouette Score :", round(score,3))

# ------------------------------------------------------------
# Cluster Summary
# ------------------------------------------------------------

cluster_summary = rfm.groupby("Cluster").agg({
    "Recency":"mean",
    "Frequency":"mean",
    "Monetary":"mean"
})

print("\nCluster Summary")
print(cluster_summary)

# ------------------------------------------------------------
# Assign Segment Names
# ------------------------------------------------------------

segment_names = {
    0: "Loyal Customers",
    1: "High Value Customers",
    2: "At Risk Customers",
    3: "New Customers"
}

rfm["CustomerSegment"] = rfm["Cluster"].map(segment_names)

print("\nCustomer Segments")
print(rfm.head())

# ------------------------------------------------------------
# Number of Customers in Each Segment
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

rfm["CustomerSegment"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Segments")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Monetary Value by Segment
# ------------------------------------------------------------

segment_sales = (
    rfm.groupby("CustomerSegment")["Monetary"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8,5))

segment_sales.plot(kind="bar")

plt.title("Average Monetary Value by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Spending")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Frequency by Segment
# ------------------------------------------------------------

segment_frequency = (
    rfm.groupby("CustomerSegment")["Frequency"]
    .mean()
)

plt.figure(figsize=(8,5))

segment_frequency.plot(kind="bar")

plt.title("Average Purchase Frequency by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Frequency")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Recency by Segment
# ------------------------------------------------------------

segment_recency = (
    rfm.groupby("CustomerSegment")["Recency"]
    .mean()
)

plt.figure(figsize=(8,5))

segment_recency.plot(kind="bar")

plt.title("Average Recency by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Days Since Last Purchase")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Scatter Plot (Frequency vs Monetary)
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="CustomerSegment",
    palette="Set2",
    s=100
)

plt.title("Customer Segments (Frequency vs Monetary)")

plt.show()

# ------------------------------------------------------------
# Scatter Plot (Recency vs Monetary)
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Monetary",
    hue="CustomerSegment",
    palette="Set1",
    s=100
)

plt.title("Customer Segments (Recency vs Monetary)")

plt.show()

# ------------------------------------------------------------
# Save Segmented Customers
# ------------------------------------------------------------

rfm.to_csv(
    "customer_segments.csv"
)

print("\nCustomer Segments Saved Successfully.")

print("="*60)
print("PART 4 COMPLETED SUCCESSFULLY")
print("="*60)

# ============================================================
# PART 5 : CUSTOMER PURCHASE PREDICTION
# ============================================================

print("\n" + "="*60)
print("PART 5 : CUSTOMER PURCHASE PREDICTION")
print("="*60)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# Create Target Variable
# ------------------------------------------------------------

# Customers with above-median spending are labeled as High Value (1)
median_spending = rfm["Monetary"].median()

rfm["HighValue"] = (
    rfm["Monetary"] >= median_spending
).astype(int)

# ------------------------------------------------------------
# Feature Selection
# ------------------------------------------------------------

X = rfm[["Recency", "Frequency", "Monetary"]]

y = rfm["HighValue"]

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))

# ------------------------------------------------------------
# Build Random Forest Model
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy : {:.2f}%".format(accuracy * 100))

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Predict for All Customers
# ------------------------------------------------------------

rfm["PredictedHighValue"] = model.predict(X)

# ------------------------------------------------------------
# Prediction Distribution
# ------------------------------------------------------------

prediction_count = (
    rfm["PredictedHighValue"]
    .value_counts()
)

plt.figure(figsize=(6,5))

prediction_count.plot(
    kind="bar"
)

plt.title("Predicted Customer Categories")
plt.xlabel("Prediction")
plt.ylabel("Number of Customers")

plt.xticks(
    [0,1],
    ["Normal","High Value"],
    rotation=0
)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Save Prediction Results
# ------------------------------------------------------------

rfm.to_csv(
    "customer_prediction_results.csv"
)

print("\nPrediction Results Saved Successfully.")

print("="*60)
print("PART 5 COMPLETED SUCCESSFULLY")
print("="*60)

# ============================================================
# PART 6 : FINAL VISUALIZATIONS & EXPORT
# ============================================================

print("\n" + "="*60)
print("PART 6 : FINAL REPORTS")
print("="*60)

import joblib

# ------------------------------------------------------------
# Customer Segment Count
# ------------------------------------------------------------

segment_count = rfm["CustomerSegment"].value_counts()

plt.figure(figsize=(8,5))

segment_count.plot(kind="bar")

plt.title("Number of Customers in Each Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Customer Count")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Customer Segment Percentage
# ------------------------------------------------------------

plt.figure(figsize=(7,7))

plt.pie(
    segment_count,
    labels=segment_count.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Customer Segment Distribution")

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Average Monetary by Segment
# ------------------------------------------------------------

avg_monetary = (
    rfm.groupby("CustomerSegment")["Monetary"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9,5))

avg_monetary.plot(kind="bar")

plt.title("Average Spending by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Monetary Value")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Average Frequency by Segment
# ------------------------------------------------------------

avg_frequency = (
    rfm.groupby("CustomerSegment")["Frequency"]
    .mean()
)

plt.figure(figsize=(9,5))

avg_frequency.plot(kind="bar")

plt.title("Average Purchase Frequency")
plt.xlabel("Customer Segment")
plt.ylabel("Average Frequency")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Average Recency by Segment
# ------------------------------------------------------------

avg_recency = (
    rfm.groupby("CustomerSegment")["Recency"]
    .mean()
)

plt.figure(figsize=(9,5))

avg_recency.plot(kind="bar")

plt.title("Average Recency")
plt.xlabel("Customer Segment")
plt.ylabel("Average Days")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Cluster-wise Summary
# ------------------------------------------------------------

summary = rfm.groupby("CustomerSegment").agg({
    "Recency":"mean",
    "Frequency":"mean",
    "Monetary":"mean"
}).round(2)

print("\nCluster Summary")
print(summary)

# ------------------------------------------------------------
# Save Summary CSV
# ------------------------------------------------------------

summary.to_csv("customer_segment_summary.csv")

# ------------------------------------------------------------
# Save Complete Customer Data
# ------------------------------------------------------------

rfm.to_csv(
    "final_customer_segments.csv",
    index=True
)

# ------------------------------------------------------------
# Save Machine Learning Model
# ------------------------------------------------------------

joblib.dump(
    model,
    "customer_prediction_model.pkl"
)

print("\nMachine Learning Model Saved Successfully.")

# ------------------------------------------------------------
# Export to Excel
# ------------------------------------------------------------

with pd.ExcelWriter("Customer_Segmentation_Report.xlsx") as writer:

    rfm.to_excel(
        writer,
        sheet_name="Customer Segments"
    )

    summary.to_excel(
        writer,
        sheet_name="Segment Summary"
    )

print("\nExcel Report Generated Successfully.")

# ------------------------------------------------------------
# Final KPIs
# ------------------------------------------------------------

print("\n" + "="*60)

print("PROJECT SUMMARY")

print("="*60)

print("Total Customers :", rfm.shape[0])

print("Total Segments :", rfm["CustomerSegment"].nunique())

print("Average Monetary Value : {:.2f}".format(
    rfm["Monetary"].mean()
))

print("Average Frequency : {:.2f}".format(
    rfm["Frequency"].mean()
))

print("Average Recency : {:.2f}".format(
    rfm["Recency"].mean()
))

print("Prediction Accuracy : {:.2f}%".format(
    accuracy * 100
))

print("="*60)

print("FILES GENERATED")

print("="*60)

print("✓ clean_customer_data.csv")
print("✓ rfm_customer_data.csv")
print("✓ customer_segments.csv")
print("✓ customer_prediction_results.csv")
print("✓ customer_segment_summary.csv")
print("✓ final_customer_segments.csv")
print("✓ Customer_Segmentation_Report.xlsx")
print("✓ customer_prediction_model.pkl")

print("="*60)
print("CUSTOMER SEGMENTATION PROJECT COMPLETED SUCCESSFULLY")
print("="*60)