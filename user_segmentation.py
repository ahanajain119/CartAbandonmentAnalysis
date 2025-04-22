from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pandas as pd

class UserSegmentation:
    def __init__(self):
        self.customer_segments = {}
        self.segment_offers = {
            'new_customer': {
                'welcome_discount': 0.15,  # 15% off
                'free_shipping': True,
                'valid_days': 7
            },
            'returning_customer': {
                'loyalty_points': 100,
                'special_discount': 0.10,  # 10% off
                'valid_days': 14
            },
            'hesitant_shopper': {
                'urgency_discount': 0.20,  # 20% off
                'limited_time': True,
                'valid_days': 3
            },
            'loyal_customer': {
                'bronze': {
                    'discount': 0.10,  # 10% off
                    'free_shipping': False,
                    'valid_days': 15
                },
                'silver': {
                    'discount': 0.15,  # 15% off
                    'free_shipping': True,
                    'valid_days': 20
                },
                'gold': {
                    'discount': 0.20,  # 20% off
                    'free_shipping': True,
                    'priority_support': True,
                    'valid_days': 25
                },
                'platinum': {
                    'discount': 0.25,  # 25% off
                    'free_shipping': True,
                    'priority_support': True,
                    'early_access': True,
                    'valid_days': 30
                }
            }
        }
    
    def analyze_customer_behavior(self, customer_data):
        # Analyze customer behavior patterns to determine their segment using K-means clustering
        # Define features for clustering
        features = [
            'No_Page_Viewed',
            'No_Cart_Viewed',
            'No_Items_Added_InCart',
            'No_Checkout_Initiated',
            'Cart_Abandoned'
        ]
        
        # Prepare data for clustering
        X = customer_data[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering with 4 clusters
        kmeans = KMeans(n_clusters=4, random_state=42)
        customer_data['Segment'] = kmeans.fit_predict(X_scaled)
        
        # Map clusters to customer segments
        segment_mapping = {
            0: 'new_customer',
            1: 'returning_customer',
            2: 'hesitant_shopper',
            3: 'loyal_customer'
        }
        
        customer_data['Customer_Segment'] = customer_data['Segment'].map(segment_mapping)
        
        # Determine loyalty tier for loyal customers
        customer_data['Loyalty_Tier'] = customer_data.apply(
            lambda x: self._determine_loyalty_tier(x) if x['Customer_Segment'] == 'loyal_customer' else None, 
            axis=1
        )
        
        return customer_data
    
    def _determine_loyalty_tier(self, customer):
        # Calculate loyalty score based on behavior with adjusted weights
        # Increased weight for successful checkouts and completed purchases
        loyalty_score = (
            customer['No_Page_Viewed'] * 0.05 +  
            customer['No_Cart_Viewed'] * 0.15 +   
            customer['No_Items_Added_InCart'] * 0.25 +  
            customer['No_Checkout_Initiated'] * 0.55     
        ) * (1 - customer['Cart_Abandoned'])  # Penalty for abandoned carts
        
        # Lowered thresholds for higher tiers
        if loyalty_score >= 10:  
            return 'platinum'
        elif loyalty_score >= 8:  
            return 'gold'
        elif loyalty_score >= 6:  
            return 'silver'
        else:
            return 'bronze' 
        