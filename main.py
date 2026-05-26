import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read dataset
data = pd.read_csv("Mall_Customers.csv")

# Select columns
x = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Create model
kmeans = KMeans(n_clusters=5, random_state=0)

# Predict clusters
data['Cluster'] = kmeans.fit_predict(x)

# Graph
plt.scatter(x.iloc[:,0], x.iloc[:,1], c=data['Cluster'])

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()

print(data.head())