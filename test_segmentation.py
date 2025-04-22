import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from user_segmentation import UserSegmentation

# Load the cleaned data
df = pd.read_csv("cleaned_cart_abandonment_data.csv")
df.set_index('ID', inplace=True)

# Initialize the segmentation system
segmentation = UserSegmentation()

# Analyze customer behavior and assign segments
df_with_segments = segmentation.analyze_customer_behavior(df)

# Print segment distribution
print("\nCustomer Segment Distribution:")
print(df_with_segments['Customer_Segment'].value_counts())

# Print loyalty tier distribution for loyal customers
print("\nLoyalty Tier Distribution:")
loyal_customers = df_with_segments[df_with_segments['Customer_Segment'] == 'loyal_customer']
print(loyal_customers['Loyalty_Tier'].value_counts())

# Analyze why customers aren't being classified as loyal
print("\nAnalyzing Customer Behavior:")
print("\nAverage values for each segment:")
print(df_with_segments.groupby('Customer_Segment')[['No_Page_Viewed', 'No_Cart_Viewed', 'No_Items_Added_InCart', 'No_Checkout_Initiated', 'Cart_Abandoned']].mean().round(2))

# Visualize segment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df_with_segments, x='Customer_Segment')
plt.title('Customer Segment Distribution')
plt.xlabel('Customer Segment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize loyalty tier distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=loyal_customers, x='Loyalty_Tier')
plt.title('Loyalty Tier Distribution')
plt.xlabel('Loyalty Tier')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Show example personalized offers
print("\nExample Personalized Offers:")
for customer_id in df_with_segments.index[:3]:
    segment = df_with_segments.loc[customer_id, 'Customer_Segment']
    tier = df_with_segments.loc[customer_id, 'Loyalty_Tier']
    print(f"\nCustomer {customer_id}:")
    print(f"Segment: {segment}")
    if segment == 'loyal_customer':
        print(f"Tier: {tier}")
        print(f"Offers: {segmentation.segment_offers[segment][tier]}")
    else:
        print(f"Offers: {segmentation.segment_offers[segment]}") 