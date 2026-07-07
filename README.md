# CUSTOMER-SEGMENTATION-AND-PURCHASE-PREDICTION-PROJECT
Customer Segmentation and Purchase Prediction using Machine Learning. This project analyzes retail customer behavior using Python, RFM Analysis, K-Means Clustering, and Random Forest. It includes data cleaning, EDA, customer segmentation, predictive modeling, visualizations, and automated report generation for business insights.
# 🛍️ Customer Segmentation and Purchase Prediction using Machine Learning

## 📌 Project Overview

This project is an end-to-end **Customer Analytics** solution developed using the **Online Retail Dataset**. It focuses on analyzing customer purchasing behavior, segmenting customers based on RFM (Recency, Frequency, Monetary) analysis, and predicting high-value customers using Machine Learning.

The project demonstrates the complete Data Analytics workflow including **Data Cleaning, Exploratory Data Analysis (EDA), Feature Engineering, Customer Segmentation, Machine Learning, and Business Reporting**.

---

## 🎯 Objectives

- Clean and preprocess retail transaction data
- Perform Exploratory Data Analysis (EDA)
- Analyze customer purchasing behavior
- Calculate RFM metrics
- Segment customers using K-Means Clustering
- Predict High-Value Customers using Random Forest
- Generate business insights through visualizations
- Export reports for business decision-making

---

## 📂 Dataset

- **Dataset:** OnlineRetail.csv
- **Domain:** Retail / E-Commerce
- **Records:** 500K+ Transactions
- **Features:** CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice, Country, Product Description

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📊 Project Workflow

### 1️⃣ Data Collection

- Load Online Retail Dataset
- Optional SQL Server Integration

### 2️⃣ Data Cleaning

- Remove Missing Values
- Remove Duplicate Records
- Handle Invalid Transactions
- Remove Outliers (IQR Method)

### 3️⃣ Feature Engineering

- Total Revenue
- Customer Lifetime Value
- Purchase Frequency
- Revenue Per Order
- Invoice Month
- Invoice Year
- Weekday Analysis

### 4️⃣ Exploratory Data Analysis

Generated multiple business visualizations including:

- Sales Distribution
- Quantity Distribution
- Unit Price Distribution
- Monthly Sales Trend
- Country-wise Sales
- Top Products
- Top Customers
- Weekday Sales Analysis
- Correlation Heatmap
- Scatter Plots
- Pair Plot
- Yearly Sales Trend

---

## 📈 RFM Analysis

Customer behavior is analyzed using:

- **Recency**
- **Frequency**
- **Monetary Value**

Customers are assigned RFM scores for business analysis.

---

## 👥 Customer Segmentation

K-Means Clustering is applied to divide customers into:

- ⭐ High Value Customers
- 🤝 Loyal Customers
- ⚠️ At Risk Customers
- 🆕 New Customers

The clustering quality is evaluated using the **Silhouette Score**.

---

## 🤖 Machine Learning

Random Forest Classifier is used to predict High-Value Customers.

### Model Evaluation

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance

---

## 📊 Visualizations

The project automatically generates **32+ professional visualizations**, including:

- Histograms
- Bar Charts
- Line Charts
- Scatter Plots
- Heatmaps
- Pair Plots
- Pie Charts
- Customer Segment Analysis
- Feature Importance
- Confusion Matrix

---

## 📁 Output Files

The project generates the following outputs:

- clean_customer_data.csv
- rfm_customer_data.csv
- customer_segments.csv
- customer_prediction_results.csv
- customer_segment_summary.csv
- final_customer_segments.csv
- Customer_Segmentation_Report.xlsx
- customer_prediction_model.pkl

---

## 📂 Project Structure

```
Customer-Segmentation-Project
│
├── OnlineRetail.csv
├── customer_segmentation.py
├── README.md
│
├── graphs/
│   ├── 32 Visualization Images
│
├── clean_customer_data.csv
├── rfm_customer_data.csv
├── customer_segments.csv
├── customer_prediction_results.csv
├── customer_segment_summary.csv
├── final_customer_segments.csv
├── Customer_Segmentation_Report.xlsx
├── customer_prediction_model.pkl
```

---

## 🚀 Key Features

- Complete Data Analytics Pipeline
- Data Cleaning & Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- RFM Customer Analysis
- Customer Segmentation
- Machine Learning Prediction
- Business KPI Analysis
- Automated Report Generation
- Professional Data Visualization

---

## 📷 Project Screenshots

> Add your **32 graph screenshots** inside the `graphs` folder and include selected images below.
><img width="800" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/9bd17078-0b5b-4c1c-b353-dc03e1ecca27" />
<img width="800" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/8486aaaf-2829-4229-90eb-1ac87eeec9de" />
<img width="800" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/7666d005-7a7f-41a3-b01c-0c596ba733c9" />
<img width="1200" height="600" alt="Figure_4" src="https://github.com/user-attachments/assets/133f88d3-c5ad-4121-b6ff-5eeabd082a7e" />
<img width="1000" height="500" alt="Figure_5" src="https://github.com/user-attachments/assets/81f7e554-6afa-4d69-bf68-ab9b15571ee6" />
<img width="1200" height="600" alt="Figure_6" src="https://github.com/user-attachments/assets/4f7c6ae3-7d2a-451a-b119-0290c470968a" />
<img width="1200" height="600" alt="Figure_7" src="https://github.com/user-attachments/assets/b0680a1b-e34a-44da-8314-61e3223a0031" />
<img width="1000" height="500" alt="Figure_8" src="https://github.com/user-attachments/assets/24f18aa6-4d25-440e-8715-b14375881d1a" />
<img width="800" height="600" alt="Figure_9" src="https://github.com/user-attachments/assets/d548d69e-f951-4de5-8b85-250b9fe935a9" />
<img width="1200" height="600" alt="Figure_10" src="https://github.com/user-attachments/assets/16ce6cb2-58b6-46dd-8684-648037c3dd3d" />
<img width="800" height="600" alt="Figure_11" src="https://github.com/user-attachments/assets/61eb03f4-075d-450d-aa12-4ac7a277d3ed" />
<img width="750" height="646" alt="Figure_12" src="https://github.com/user-attachments/assets/094ab327-c017-4b95-8e0f-ec3f88812306" />
<img width="800" height="500" alt="Figure_13" src="https://github.com/user-attachments/assets/d773d684-1767-4049-8022-c6d2a8919c28" />
<img width="800" height="500" alt="Figure_14" src="https://github.com/user-attachments/assets/1cfc915b-ddb7-41b3-adb2-d252bec4fe2e" />
<img width="800" height="500" alt="Figure_15" src="https://github.com/user-attachments/assets/ac289867-f190-4409-8f48-f27c66f0ae93" />
<img width="800" height="500" alt="Figure_16" src="https://github.com/user-attachments/assets/da380d62-b912-4aad-a91c-56d138ac1668" />
<img width="1200" height="600" alt="Figure_17" src="https://github.com/user-attachments/assets/465d33cc-c7de-467b-a003-7ded7c289b60" />
<img width="1200" height="600" alt="Figure_18" src="https://github.com/user-attachments/assets/f47234a6-5932-4e05-8566-723de1257ef4" />
<img width="1366" height="655" alt="Figure_19" src="https://github.com/user-attachments/assets/13e3bfd4-daa7-4a5f-a496-db3ff8dedc58" />
<img width="1366" height="655" alt="Figure_20" src="https://github.com/user-attachments/assets/d17d01d0-4589-4772-926e-d0936da20d59" />
<img width="800" height="500" alt="Figure_21" src="https://github.com/user-attachments/assets/45fd932f-d193-498e-975d-2242145b69d1" />
<img width="800" height="500" alt="Figure_22" src="https://github.com/user-attachments/assets/3124756d-6fdf-4b7d-a872-72f2047ae8a7" />
<img width="800" height="500" alt="Figure_23" src="https://github.com/user-attachments/assets/e509e0f0-8960-4c36-852b-cde2abcb4527" />
<img width="800" height="500" alt="Figure_24" src="https://github.com/user-attachments/assets/8936b130-a8c6-4b17-8d67-2049099f273d" />
<img width="1000" height="600" alt="Figure_25" src="https://github.com/user-attachments/assets/9787ada8-ca8b-4b8c-9973-f8eb118e0a63" />
<img width="1000" height="600" alt="Figure_26" src="https://github.com/user-attachments/assets/07a5452f-dbb4-4e9e-9a6d-8c6e4419fe8c" />
<img width="600" height="500" alt="Figure_27" src="https://github.com/user-attachments/assets/3ab0309e-c050-4dab-8e48-f9d0c7606ea2" />
<img width="800" height="500" alt="Figure_28" src="https://github.com/user-attachments/assets/d376efb0-20b8-4257-b39b-0d559bba33f9" />
<img width="800" height="500" alt="Figure_29" src="https://github.com/user-attachments/assets/09b28c53-e20e-4b7b-bc99-88cf49cb4262" />
<img width="600" height="500" alt="Figure_30" src="https://github.com/user-attachments/assets/672dd02c-5e76-4b7a-b73f-5dcfcd0b15a3" />
<img width="800" height="500" alt="Figure_31" src="https://github.com/user-attachments/assets/72ca886a-0eac-4b21-8169-618a353d613a" />
<img width="1366" height="655" alt="Figure_33" src="https://github.com/user-attachments/assets/e29f976b-827c-4479-8bec-4a8bcc8a7b45" />
<img width="900" height="500" alt="Figure_34" src="https://github.com/user-attachments/assets/12bef300-6554-41b4-a2ac-a3dfa68b52e8" />
<img width="900" height="500" alt="Figure_35" src="https://github.com/user-attachments/assets/5dd454e3-86b3-4a99-a195-fb8a1bdad19f" />
<img width="900" height="500" alt="Figure_36" src="https://github.com/user-attachments/assets/54e3ca43-236d-4fc0-9fba-5def20066ac2" />

Example:

```markdown
![Sales Distribution](graphs/01_sales_distribution.png)

![Customer Segmentation](graphs/20_customer_segments.png)

![Confusion Matrix](graphs/28_confusion_matrix.png)
```

---

## 💼 Skills Demonstrated

- Python Programming
- Data Cleaning
- Data Analysis
- Exploratory Data Analysis (EDA)
- Data Visualization
- Machine Learning
- Customer Segmentation
- Predictive Analytics
- Business Intelligence
- Feature Engineering
- K-Means Clustering
- Random Forest Classification

---

## 📌 Future Improvements

- Interactive Power BI Dashboard
- Streamlit Web Application
- SQL Database Integration
- Real-Time Prediction API
- Customer Recommendation System

---

## 👨‍💻 Author

**Pratap Reddy**

Aspiring **Data Analyst** passionate about transforming raw data into actionable business insights using **Python, SQL, Excel, Power BI, and Machine Learning**.

---

## ⭐ If you found this project useful, don't forget to Star this repository!
