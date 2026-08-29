import pandas as pd
df = pd.read_csv('e commerce dataset (1).csv', encoding='latin1')
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nStatistics:")
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,5))
sns.histplot(df['Sales'], bins=50, kde=True)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.savefig('sales_distribution.png')
plt.show()
plt.figure(figsize=(8,5))
sns.barplot(x='Region', y='Sales', data=df, estimator=sum)
plt.title('Total Sales by Region')
plt.savefig('sales_by_region.png')
plt.show()
plt.figure(figsize=(8,5))
sns.barplot(x='Category', y='Profit', data=df, estimator=sum)
plt.title('Total Profit by Category')
plt.savefig('profit_by_category.png')
plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
customer_data = df.groupby('Customer Name').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['Sales', 'Profit']])
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Segment'] = kmeans.fit_predict(scaled_data)
print(customer_data.head(10))
plt.figure(figsize=(8,6))
plt.scatter(customer_data['Sales'], customer_data['Profit'], c=customer_data['Segment'], cmap='viridis')
plt.xlabel('Total Sales')
plt.ylabel('Total Profit')
plt.title('Customer Segmentation')
plt.savefig('customer_segmentation.png')
plt.show()
