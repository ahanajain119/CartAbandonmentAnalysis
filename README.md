## Cart Abandonment Analysis & Prediction

Predict cart abandonment risk and segment users using a Multilayer Perceptron (Keras/TensorFlow) and KMeans clustering. A Streamlit UI is planned and currently under development.

### Features
- **Data cleaning & feature engineering**: handles missing values, derives `Checkout_Conversion_Rate`, `Browsing_Intensity`.
- **Multilayer Perceptron (MLP)**: neural network for binary prediction of `Cart_Abandoned`.
- **User segmentation**: KMeans (k=4) over behavioral features; loyalty tiering for loyal customers.
- **Evaluation & visuals**: correlation heatmap, confusion matrix, several exploratory plots.
- **Streamlit app (in progress)**: planned UI to upload CSV and run predictions.

### Tech Stack
- **Python**
- **Libraries**: pandas, numpy, scikit-learn, TensorFlow/Keras, seaborn, matplotlib, joblib, Streamlit (planned)

### Repository Structure
```
CartAbandonAnalysis/
  cart_abandonment_analysis.py     # training, EDA, model save, segmentation integration
  checkout_analysis.py             # checkout funnel metrics helpers
  cleaned_cart_abandonment_data.csv
  data_cart_abandonment.csv        # raw dataset (sample/input)
  mnn_model.h5                     # saved Keras model (generated)
  scaler.pkl                       # saved StandardScaler (generated)
  streamlit_app.py                 # Streamlit UI for prediction
  user_segmentation.py             # KMeans-based segmentation + loyalty tiering
```

### Data Schema (expected columns)
Required columns used across training and segmentation:
- `ID`
- `Is_Product_Details_viewed` (Yes/No)
- `Session_Activity_Count`
- `No_Page_Viewed`, `No_Cart_Viewed`, `No_Items_Added_InCart`, `No_Items_Removed_FromCart`
- `No_Checkout_Initiated`, `No_Checkout_Confirmed`
- `Cart_Abandoned` (0/1 target)

Derived features (computed in training script):
- `Checkout_Conversion_Rate = No_Checkout_Confirmed / (No_Checkout_Initiated + 1)`
- `Browsing_Intensity = No_Page_Viewed / (Session_Activity_Count + 1)`

### Modeling (MLP)
- **Inputs**
  - Browsing: `[No_Page_Viewed, Browsing_Intensity]`
  - Cart: `[No_Cart_Viewed, No_Items_Added_InCart, No_Items_Removed_FromCart]`
  - Checkout: `[No_Checkout_Initiated, No_Checkout_Confirmed]`
- **Architecture**
  - Multilayer Perceptron over engineered features with hidden `Dense(8, relu)` layers and final `Dense(1, sigmoid)`
- **Training**
  - Loss: `binary_crossentropy`, Optimizer: `adam`, Metrics: `accuracy`
  - Train/validation split via `train_test_split`
  - Standardization with `StandardScaler` (saved to `scaler.pkl`)
  - Model saved to `mnn_model.h5`
- **Metrics**
  - Accuracy, Precision, Recall, F1; Confusion matrix plotted

### Segmentation
- **Algorithm**: scikit-learn `KMeans(n_clusters=4, random_state=42)`
- **Clustering features**: `No_Page_Viewed`, `No_Cart_Viewed`, `No_Items_Added_InCart`, `No_Checkout_Initiated`, `Cart_Abandoned`
- **Segment mapping**: `{0: new_customer, 1: returning_customer, 2: hesitant_shopper, 3: loyal_customer}`
- **Loyalty tiering** (for `loyal_customer`): `bronze`, `silver`, `gold`, `platinum` based on a weighted loyalty score emphasizing successful checkouts

### Setup
1) Create and activate a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate    # Windows
```

2) Install dependencies
```bash
pip install -U pandas numpy scikit-learn tensorflow seaborn matplotlib joblib streamlit
```

### Train and Evaluate
Runs EDA, computes features, trains the MLP, saves `scaler.pkl` and `mnn_model.h5`, performs segmentation-driven analysis and plots.
```bash
python cart_abandonment_analysis.py
```
Outputs:
- `cleaned_cart_abandonment_data.csv`
- `scaler.pkl`
- `mnn_model.h5`
- Console metrics and plots

### Streamlit App (planned / in progress)
- Status: Not yet integrated; `streamlit_app.py` is a work-in-progress.
- Intended behavior: Upload CSV → scale with saved `scaler.pkl` → predict via `mnn_model.h5` → display results.
- When integration is completed, you'll launch with:
  ```bash
  streamlit run streamlit_app.py
  ```
  and follow the on-screen instructions.

### Notes
- If you change features, keep the same `StandardScaler` used during training when serving predictions.
- Ensure column names match exactly (case/spacing) to avoid preprocessing errors.
- For segmentation-only workflows, call `UserSegmentation().analyze_customer_behavior(df)` from `user_segmentation.py`.

### License
For academic/educational use. Adjust as needed for your project.
