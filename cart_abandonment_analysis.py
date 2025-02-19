from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

import pandas as pd

df = pd.read_csv("data_cart_abandonment.csv")  

print(df.info())
print(df.head())

df.fillna(0, inplace=True)
df.columns = df.columns.str.strip()  # Removes unwanted spaces
df['Is_Product_Details_viewed'] = df['Is_Product_Details_viewed'].map({'Yes': 1, 'No': 0})
df['Checkout_Conversion_Rate'] = df['No_Checkout_Confirmed'] / (df['No_Checkout_Initiated'] + 1)  # Avoid division by zero
df['Browsing_Intensity'] = df['No_Page_Viewed'] / (df['Session_Activity_Count'] + 1)

df.to_csv("cleaned_cart_abandonment_data.csv", index=False)

print(df.describe())

import seaborn as sns
import matplotlib.pyplot as plt

#Plot Cart Abandonment Rate
sns.countplot(x=df["Cart_Abandoned"])
plt.title("Cart Abandonment Distribution")
plt.xlabel("Abandoned (1) vs Completed (0)")
plt.ylabel("Count")
plt.show()

#Check Correlation Between Features
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#Analyze Checkout Steps
sns.scatterplot(x=df["No_Checkout_Initiated"], y=df["No_Checkout_Confirmed"])
plt.title("Checkouts Initiated vs Confirmed")
plt.xlabel("Checkouts Initiated")
plt.ylabel("Checkouts Confirmed")
plt.show()

#Study Browsing Behavior
sns.boxplot(x=df["Cart_Abandoned"], y=df["No_Page_Viewed"])
plt.title("Pages Viewed vs Cart Abandonment")
plt.xlabel("Cart Abandoned (1) vs Completed (0)")
plt.ylabel("Number of Pages Viewed")
plt.show()

#y is target and x is features
# Drop 'ID' as it's just an identifier, and separate target variable
X = df.drop(columns=['ID', 'Cart_Abandoned'])
y = df['Cart_Abandoned']

#Split the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#Build the Modular Neural Network Model

#Preprocess Data & Split into Modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("cleaned_cart_abandonment_data.csv")

# Define feature groups for different modules
browsing_features = ["No_Page_Viewed", "Browsing_Intensity"]
cart_features = ["No_Cart_Viewed", "No_Items_Added_InCart", "No_Items_Removed_FromCart"]
checkout_features = ["No_Checkout_Initiated", "No_Checkout_Confirmed"]

# Target variable
target = "Cart_Abandoned"

# Normalize features
scaler = StandardScaler()
df[browsing_features + cart_features + checkout_features] = scaler.fit_transform(df[browsing_features + cart_features + checkout_features])

# Split dataset into training and testing sets
X_browsing = df[browsing_features]
X_cart = df[cart_features]
X_checkout = df[checkout_features]
y = df[target]

X_b_train, X_b_test, X_c_train, X_c_test, X_ch_train, X_ch_test, y_train, y_test = train_test_split(
    X_browsing, X_cart, X_checkout, y, test_size=0.2, random_state=42
)

print("Browsing Features:", browsing_features)

# Browsing Behavior Module
input_browsing = Input(shape=(len(browsing_features),))
browsing_layer = Dense(8, activation="relu")(input_browsing)

# Cart Interaction Module
input_cart = Input(shape=(len(cart_features),), name="Cart_Input")
cart_layer = Dense(8, activation="relu")(input_cart)

# Checkout Behavior Module
input_checkout = Input(shape=(len(checkout_features),), name="Checkout_Input")
checkout_layer = Dense(8, activation="relu")(input_checkout)

# Merge All Modules
merged = Concatenate()([browsing_layer, cart_layer, checkout_layer])

# Final Decision Layer
hidden_layer = Dense(8, activation="relu")(merged)
output = Dense(1, activation="sigmoid", name="Output_Layer")(hidden_layer)

# Define the Modular Model
model = Model(inputs=[input_browsing, input_cart, input_checkout], outputs=output)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Train the Model
history = model.fit(
    [X_b_train, X_c_train, X_ch_train], y_train,
    validation_data=([X_b_test, X_c_test, X_ch_test], y_test),
    epochs=50,  # Adjust epochs if needed
    batch_size=32
)

import numpy as np

# Get model predictions (probabilities)
y_pred_prob = model.predict([X_b_test, X_c_test, X_ch_test])

# Convert probabilities to binary predictions (0 or 1)
y_pred = (y_pred_prob > 0.5).astype(int)

# Print some sample predictions
print("Actual:", y_test[:10].values)  # First 10 actual labels
print("Predicted:", y_pred[:10].flatten())  # First 10 predicted labels

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Completed (0)", "Abandoned (1)"], yticklabels=["Completed (0)", "Abandoned (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Find misclassified examples
misclassified_indices = np.where(y_test.values != y_pred.flatten())[0]

# Show first 5 misclassified cases
print("Misclassified Examples:")
for i in misclassified_indices[:5]:
    print(f"Actual: {y_test.values[i]}, Predicted: {y_pred[i][0]}")
