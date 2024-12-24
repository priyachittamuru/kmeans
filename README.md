# kmeans_creditcard
# Credit Card Dataset for Clustering

## Overview
This dataset provides information about customers' credit card usage patterns and is designed for clustering tasks. It is ideal for exploring customer segmentation, behavior analysis, and related applications in the financial sector.

---

## Dataset Summary
- **Source:** Synthetic dataset created for educational and research purposes.
- **File Format:** CSV
- **Number of Instances (Rows):** 8,950
- **Number of Attributes (Columns):** 18

---

## Data Features
The dataset contains the following columns:

| Feature              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `Customer ID`        | Unique identifier for each customer                                         |
| `Balance`            | Average balance carried by the customer over the past six months           |
| `Balance Frequency`  | Frequency of balance updates over six months (0 to 1 scale)                |
| `Purchases`          | Total purchase amount over six months                                      |
| `One-off Purchases`  | Purchases made in single transactions                                      |
| `Installments Purchases` | Purchases made in installments                                           |
| `Cash Advance`       | Total cash advances taken by the customer                                  |
| `Purchases Frequency`| Frequency of purchases over six months (0 to 1 scale)                      |
| `One-off Purchases Frequency` | Frequency of one-off purchases over six months (0 to 1 scale)     |
| `Installments Purchases Frequency` | Frequency of installment purchases over six months (0 to 1) |
| `Cash Advance Frequency` | Frequency of cash advances over six months (0 to 1 scale)             |
| `Cash Advance Trx`   | Number of transactions involving cash advances                             |
| `Purchases Trx`      | Number of purchase transactions                                             |
| `Credit Limit`       | Credit limit assigned to the customer                                       |
| `Payments`           | Total payments made by the customer                                        |
| `Minimum Payments`   | Minimum amount paid by the customer                                        |
| `PRC Full Payment`   | Percentage of full payments made (0 to 1 scale)                            |
| `Tenure`             | Number of months the account has been active                               |

---

## Applications
The dataset can be used for:
- **Customer Segmentation:** Group customers based on their spending, payments, and balance behaviors.
- **Behavioral Insights:** Identify high spenders, installment buyers, or frequent cash advance users.
- **Clustering Analysis:** Apply clustering algorithms such as K-means or hierarchical clustering.
- **Visualization:** Explore patterns in customer behavior using dimensionality reduction and visual techniques.

---

## Preprocessing Suggestions
To prepare the dataset for analysis:
1. **Handle Missing Values:** Address missing values in columns like `Minimum Payments` and `Credit Limit`.
2. **Feature Scaling:** Normalize columns with wide value ranges, such as `Balance`, `Purchases`, and `Payments`.
3. **Feature Engineering:** Create derived features, e.g., `average purchase size` or `cash-to-purchase ratio`.
4. **Dimensionality Reduction:** Apply PCA or t-SNE for visualization and noise reduction.

---

## Usage Instructions
1. Load the dataset:
   ```python
   import pandas as pd
   df = pd.read_csv('CC GENERAL.csv')
   ```
2. Clean and preprocess the data based on the suggestions above.
3. Apply clustering algorithms such as K-means or DBSCAN.
4. Visualize the clusters to gain insights into customer segmentation.

---

## Example Code
```python
# Load the dataset
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
file_path = 'CC GENERAL.csv'
data = pd.read_csv(file_path)

# Handle missing values
data.fillna(data.median(), inplace=True)

# Select relevant features
features = data.drop(columns=['Customer ID'])

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the dataset
data['Cluster'] = clusters

# Save the clustered data
data.to_csv('clustered_data.csv', index=False)
```

---

## Notes
- The dataset is synthetic and does not represent real customer data.
- Ensure ethical use of this dataset for educational purposes only.

---

## References
- [Kaggle Dataset Page](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

