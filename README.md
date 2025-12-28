# Netflix Churn
## 📌 Project Overview
This project explores **Netflix data** to extract meaningful insights into customer behavior, sales trends, and business strategies. The analysis focuses on:

- Data preprocessing & cleaning
- Exploratory Data Analysis (EDA)
- Customer segmentation & purchasing behavior
- Price elasticity & marketing impact
- Machine learning modeling 

- 
## Architecture
![Architecture](Nerflix.png)


## 📊 Analysis & Insights

### 1️⃣ Data Cleaning & Preprocessing
- Missing values were handled appropriately to ensure data consistency.
- Data types were optimized for better performance.
- Unnecessary columns were removed for a streamlined analysis.

### 2️⃣ Exploratory Data Analysis (EDA)
##### These columns ("zip_code", "latitude", "longitude", "city") represent geographic information
##### Geographic data is not directly related to customer behavior and is unlikely to influence churn prediction
##### Removing these features reduces dimensionality and avoids adding noise to the model

## EDA

### Heatmap Analysis and Observations

The heatmap visualizes the correlation between numerical features in the dataset. Below are the key observations:

#### **1. Strong Positive Correlations**
- `total_charges`and `total_revenue` show high correlation with one another.
  - **Implication**: These features are redundant, and including all of them may lead to multicollinearity in models.
  - Consider retaining only one of them, such as `total_revenue`, which combines multiple components.

#### **2. Weak or No Correlations**
- Features like `age`, `number_of_family_dependents`, and `number_of_referrals` show weak or no correlation with most other features.
  - **Implication**: These features may not directly impact other metrics but could still play a role in churn prediction through non-linear relationships.

#### **3. Negative Correlations**
- `tenure_in_months` shows a negative correlation with features like `monthly_charge` and `total_refunds`.
  - **Implication**: Customers with longer tenure tend to have lower monthly charges or are less likely to churn, indicating loyalty.

#### **4. Neutral Relationships**
- Features such as `avg_monthly_gb_download` and `number_of_referrals` appear neutral, meaning they neither positively nor negatively correlate strongly with other metrics.
  - **Implication**: These features may have independent predictive power and should not be ignored.


## Checking for outliers

### Handling Outliers Using the IQR Method

#### **Why Exclude Certain Columns?**
The columns `number_of_family_dependents`, `total_refunds`, and `total_extra_data_charges` were excluded from the IQR-based outlier handling due to the following reasons:

1. **`number_of_family_dependents`**:
   - This column represents the count of family dependents and typically takes discrete integer values.
   - Outliers in this column are more likely to represent real scenarios (e.g., large families) and should not be capped or modified.

2. **`total_refunds`**:
   - Refund values can vary significantly between customers. High refund amounts might indicate problematic cases leading to churn, which are critical for prediction.
   - Treating high refund values as outliers could remove important information.

3. **`total_extra_data_charges`**:
   - High extra data charges likely reflect heavy data users, which are an important segment for analysis.
   - These outliers are genuine and should not be capped, as they can provide valuable insights into customer behavior.

#### **General Approach**
For other numerical columns:
- The IQR method is applied to cap values outside the range `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]` to the respective bounds.
- This ensures that extreme outliers in features like `monthly_charge` and `total_long_distance_charges` are handled while preserving data integrity.


##### Exclude 3 Columns "number_of_family_dependents", "total_refunds", "total_extra_data_charges"
##### The column contains meaningful values and is not distorted by outliers.

### Handling Missing Values

#### **1. Domain-Specific Imputation**
- For the column `internet_type`, missing values were filled with `"None"`, indicating no internet service.
- For `avg_monthly_gb_download`, missing values were filled with `0`, as customers without internet would have zero GB downloads.
- For `offer`, missing values were filled with `"None"`, meaning no offer was accepted by the customer.

**Reason**: These replacements align with the dataset's context and prevent losing important information due to missing values.

---

#### **2. Categorical Columns Related to Services**
- Columns like `online_security`, `online_backup`, `device_protection_plan`, etc., were filled with `"No"`, as missing values indicate that the customer did not subscribe to these services.

**Reason**: The data dictionary suggests missing values in these columns imply non-subscription.

---

#### **3. Remaining Categorical Features**
- Missing values in other categorical columns were replaced with `"Unknown"`.

**Reason**: For categorical features without clear domain knowledge, `"Unknown"` helps retain the data while indicating the lack of information.

---

#### **4. Numerical Columns**
- Missing values in numerical columns were replaced with the **median** of each column.

**Reason**: Median imputation is robust to outliers and preserves the central tendency of the data.

---

#### **5. One-Hot Encoding**
- Categorical columns like `gender`, `married`, `offer`, `internet_service`, etc., were converted into dummy variables using one-hot encoding with `drop_first=True`.

**Reason**: One-hot encoding is necessary for machine learning algorithms, and `drop_first=True` avoids multicollinearity.

---

#### **6. Fixing Negative Values**
- Negative values in the `monthly_charge` column were replaced with their absolute values.

**Reason**: Negative charges are likely data entry errors and should be corrected.

---

#### **7. Removing Duplicates**
- Duplicate rows were identified and removed.

**Reason**: Duplicate rows could skew analysis and model performance.

---

#### **Final Check**
- After all the imputation and cleaning steps:
  - Total missing values: **0**
  - Duplicate rows: **0**

**Conclusion**: All missing values were handled using domain-specific and statistical techniques, ensuring the dataset is clean and ready for analysis or modeling.


### Selecting and Scaling Numerical Columns

#### **Purpose of the Code**

1. **Select Numerical Columns Safely**
   - `numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns`
   - This identifies all columns with numerical data types (`int64` and `float64`) in the dataset.
   - **Reason**: It ensures only numerical columns are selected for scaling, avoiding errors from non-numerical data.

2. **Drop the Target Variable (`viewer_status`)**
   - `if 'viewer_status' in numeric_cols: numeric_cols = numeric_cols.drop('viewer_status')`
   - The target variable, `viewer_status`, is excluded from the list of numerical columns before scaling.
   - **Reason**: 
     - `viewer_status` is the label for prediction, not a feature.
     - Including it during scaling would distort its values and could lead to data leakage during modeling.

3. **Scaling Numerical Features**
   - `scaler = StandardScaler()`
   - `df_scaled = df[numeric_cols].copy()`
   - `df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])`
   - This applies **Standard Scaling** to the numerical columns, transforming the values to have a mean of 0 and a standard deviation of 1.
   - **Reason**:
     - Scaling ensures that all numerical features contribute equally to distance-based algorithms (e.g., clustering, logistic regression).
     - It also speeds up convergence for gradient-based models and avoids bias from features with larger ranges.

---

#### **Why This Approach?**
- **Dynamic Column Selection**: The code dynamically selects numerical columns, ensuring scalability to future datasets with different column names or types.
- **Safe Target Handling**: The check for `viewer_status` prevents accidental inclusion of the target variable.
- **Robust Feature Preparation**: Standard scaling is a widely accepted preprocessing step for numerical features in machine learning.



### Principal Component Analysis (PCA) Interpretation

#### **Purpose of the PCA**
- PCA was performed to reduce the dimensionality of the dataset while retaining most of the variance in the data.
- This helps in simplifying data for clustering and modeling, improving computational efficiency and interpretability.

---

#### **Explained Variance Analysis**
- The **cumulative explained variance plot** shows the proportion of variance retained by each principal component.
- A horizontal red line is drawn at **95% cumulative variance** to identify the minimum number of components required to retain most of the information.

---

#### **Key Observations**
1. **Number of Components**:
   - From the plot, approximately **8 components** are required to retain **95% of the variance** in the data.
   - This reduces the dataset's dimensionality from the original number of features while preserving most of its variability.

2. **Variance Retention**:
   - The first few components capture most of the variance:
     - **First Component**: Captures the largest proportion of variance.
     - **Subsequent Components**: Each adds diminishing amounts of variance.

3. **Dimensionality Reduction**:
   - By reducing the dataset to 8 components, we simplify the dataset for downstream tasks such as clustering or predictive modeling, without significant information loss.



#### **Conclusion**
- PCA has successfully reduced the dataset's dimensionality to 8 components while retaining 95% of the information.
- This step is critical for computational efficiency and feature simplification in further analysis.


### Why \( k = 4 \) Was Chosen Based on the Elbow Method

The Elbow Method helps determine the optimal number of clusters (\( k \)) in K-Means clustering by plotting the **inertia** (sum of squared distances to the nearest cluster center) against the number of clusters. 

#### Key Observations:
1. **Decreasing Inertia**:
   - As \( k \) increases, the inertia decreases because adding more clusters reduces the variance within each cluster.
   - However, the rate of decrease diminishes after a certain point, which is the "elbow."

2. **Elbow Point**:
   - The "elbow" is the point where the curve starts to flatten, indicating diminishing returns in variance reduction.
   - From the graph, the elbow appears at \( k = 4 \).

3. **Optimal Trade-Off**:
   - Choosing \( k = 4 \) balances simplicity (fewer clusters) with sufficient variance reduction. 
   - Beyond \( k = 4 \), additional clusters do not significantly improve the clustering quality but increase complexity.

### K-Means Clustering with \( k = 4 \): Interpretation

The scatter plot and cluster summary provide insights into the clustering results based on the first two principal components (PC1 and PC2) and the mean values of each cluster across all principal components.

#### **Scatter Plot Interpretation**
1. **Visual Segmentation**:
   - The data points are color-coded into 4 clusters, each represented by a distinct color.
   - The plot shows how the clusters are distributed based on the first two principal components (PC1 and PC2).

2. **Cluster Characteristics**:
   - Clusters exhibit significant overlap, suggesting some similarity between groups along PC1 and PC2.
   - Further analysis of additional principal components is required to fully understand their separability.

3. **Compactness**:
   - Clusters appear reasonably compact, suggesting that \( k = 4 \) was a good choice.

#### **Cluster Summary Table**
- The cluster summary shows the mean values of each cluster across the principal components (PC1 to PC8). 
- These values represent the average contribution of each principal component to the respective cluster.

---

#### **Key Observations**
1. **Cluster 0**:
   - Negative mean in PC1 and positive in PC2, indicating that members in this cluster are influenced more by PC2.
   - Could represent a group with distinct features compared to other clusters.

2. **Cluster 1**:
   - Small negative values across most PCs, indicating less variation and potentially representing a baseline group.

3. **Cluster 2**:
   - High negative values in PC2 and positive in PC6, showing significant variation compared to other clusters.

4. **Cluster 3**:
   - Positive mean in PC1 and PC6, suggesting strong influence from these components and distinctiveness from Cluster 0.

### Predictive Modeling: Logistic Regression vs. Random Forest

#### **Process**
1. **Dataset Splitting**:
   - The dataset was split into training and testing sets (80-20 split) with stratification to ensure balanced class distribution.

2. **Feature Scaling**:
   - `StandardScaler` was applied to normalize the features, which is particularly important for logistic regression.

3. **Logistic Regression**:
   - Trained using `max_iter=500` to ensure model convergence.
   - Predictions were evaluated using accuracy and ROC-AUC scores.

4. **Random Forest Classifier**:
   - A Random Forest model was trained with 100 estimators (`n_estimators=100`) and default hyperparameters.
   - Predictions were evaluated using accuracy and ROC-AUC scores.

5. **Feature Importance**:
   - Random Forest provides a feature importance metric, highlighting the most influential features for predictions.

---

#### **Model Performance**
| Metric                | Logistic Regression | Random Forest  |
|-----------------------|----------------------|----------------|
| **Accuracy**          | 85.13%              | 86.19%         |
| **ROC-AUC**           | 91.49%              | 91.83%         |

---

#### **Top 5 Predictive Features (From Random Forest)**:
| Feature                  | Importance |
|--------------------------|------------|
| Tenure in Months         | 0.124866   |
| Total Revenue            | 0.103061   |
| Total Charges            | 0.103152   |
| Monthly Charge           | 0.076580   |
| Total Long Distance Charges | 0.059908   |

- **Tenure in Months**: Customers with longer tenure are less likely to churn, indicating loyalty.
- **Total Revenue and Total Charges**: High revenue and charges may correlate with higher engagement or dissatisfaction.
- **Monthly Charge**: Indicates the customer's plan type, which might influence churn behavior.
- **Total Long Distance Charges**: Could reflect additional service usage impacting satisfaction.

---

#### **Insights**
1. **Model Comparison**:
   - Both models performed well, but Random Forest achieved slightly higher accuracy and ROC-AUC.
   - Random Forest is better suited for this dataset as it captures non-linear relationships and interactions between features.

2. **Feature Importance**:
   - The top predictive features align with business insights, highlighting engagement, tenure, and charges as critical factors in churn prediction.


### Top 5 Predictive Factors and Model Comparison

#### **Top 5 Most Predictive Factors (From Random Forest)**:
1. **Tenure in Months** (Importance: 0.124866): 
   - Customers with longer tenure are less likely to churn, indicating loyalty and satisfaction.
2. **Total Revenue** (Importance: 0.103061):
   - High revenue could correlate with higher engagement or dissatisfaction, depending on customer experience.
3. **Total Charges** (Importance: 0.103152):
   - Represents overall spending, which might influence customer churn based on perceived value.
4. **Monthly Charge** (Importance: 0.076580):
   - Reflects the type of plan or services subscribed by the customer.
5. **Total Long Distance Charges** (Importance: 0.059908):
   - Suggests additional service usage which may impact satisfaction and retention.


### Customer Segment at Most Risk for Churn and Recommendations

#### **Most at Risk Segment**
- Based on clustering, **Cluster 1** had the highest churn rate. 
- Characteristics of this segment include:
  - **Low tenure**: Customers with less time with the company are more likely to churn.
  - **Higher charges**: Customers paying higher total or monthly charges might feel dissatisfied or perceive poor value.

#### **Recommended Actions to Reduce Churn**
1. **Engagement Strategies**:
   - Offer loyalty programs or incentives to customers with low tenure to encourage retention.
   - Provide personalized offers to make them feel valued early in their journey.

2. **Pricing Optimization**:
   - Analyze if customers perceive high charges as a barrier and offer discounts or customized plans to reduce dissatisfaction.

3. **Customer Support Enhancement**:
   - Focus on improving the customer experience for this segment with proactive support and faster issue resolution.

4. **Feedback Mechanisms**:
   - Collect feedback from at-risk customers to understand their dissatisfaction points and take corrective actions.

By addressing the pain points of Cluster 1, the company can reduce churn and improve overall customer satisfaction.


### 3️⃣ Customer Segmentation & Sales Trends
- Identified customer segments based on purchase behavior.
- Analyzed the impact of discounts, promotions, and seasonal trends.
- Derived insights into customer retention and lifetime value.

### 4️⃣ Marketing & Business Strategy
- Evaluated the effectiveness of marketing campaigns.
- Measured customer response to loyalty programs.
- Assessed which products generate the highest revenue and customer engagement.

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/skandvj/Predictive-Sales-Performance-Optimization-for-Netflix/)
cd <your-repo-folder>
```

### 2️⃣ Install Dependencies
Ensure you have Python installed. Then, install required dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook
Launch Jupyter Notebook and open `main.ipynb`:
```bash
jupyter notebooksmain.ipynb
```

## 🔗 References
- Natflix dataset

---

**Maintainer:** [Skand Vijay](https://www.linkedin.com/in/skandvijay/)  

