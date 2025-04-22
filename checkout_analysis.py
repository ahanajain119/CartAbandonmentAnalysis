import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class CheckoutAnalyzer:
    def __init__(self):
        self.checkout_steps = [
            'cart_view',
            'checkout_initiation',
            'form_filling',
            'payment_selection',
            'order_review',
            'order_confirmation'
        ]
        
    def analyze_checkout_process(self, df):
        # Calculate drop-off rates between steps
        drop_off_rates = self._calculate_drop_off_rates(df)
        
        # Identify patterns in abandoned checkouts
        abandonment_patterns = self._analyze_abandonment_patterns(df)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(drop_off_rates, abandonment_patterns)
        
        return {
            'drop_off_rates': drop_off_rates,
            'abandonment_patterns': abandonment_patterns,
            'suggestions': suggestions
        }
    
    def _calculate_drop_off_rates(self, df):
        # Calculate conversion rates between steps
        rates = {}
        
        # Cart View to Checkout Initiation
        rates['cart_to_checkout'] = df['No_Checkout_Initiated'].sum() / df['No_Cart_Viewed'].sum()
        
        # Checkout Initiation to Confirmation
        rates['checkout_to_confirmation'] = df['No_Checkout_Confirmed'].sum() / df['No_Checkout_Initiated'].sum()
        
        # Overall checkout completion rate
        rates['overall_completion'] = df['No_Checkout_Confirmed'].sum() / df['No_Cart_Viewed'].sum()
        
        return rates
    
    def _analyze_abandonment_patterns(self, df):
        patterns = {}
        
        # Analyze form filling behavior
        patterns['form_filling'] = {
            'avg_time': df['Session_Activity_Count'].mean(),
            'completion_rate': df['No_Checkout_Confirmed'].sum() / df['No_Checkout_Initiated'].sum()
        }
        
        # Analyze payment behavior
        patterns['payment'] = {
            'success_rate': df['No_Checkout_Confirmed'].sum() / df['No_Checkout_Initiated'].sum()
        }
        
        # Identify common abandonment points
        patterns['abandonment_points'] = self._identify_abandonment_points(df)
        
        return patterns
    
    def _identify_abandonment_points(self, df):
        points = {}
        
        # Calculate abandonment at each step
        points['cart_view'] = df['No_Cart_Viewed'].sum() - df['No_Checkout_Initiated'].sum()
        points['checkout_initiation'] = df['No_Checkout_Initiated'].sum() - df['No_Checkout_Confirmed'].sum()
        
        return points
    
    def _generate_optimization_suggestions(self, drop_off_rates, patterns):
        suggestions = []
        
        # Analyze cart to checkout conversion
        if drop_off_rates['cart_to_checkout'] < 0.3:
            suggestions.append({
                'step': 'Cart to Checkout',
                'issue': 'Low conversion from cart to checkout',
                'suggestions': [
                    'Add a prominent checkout button',
                    'Implement a mini-cart preview',
                    'Show shipping costs early in the process'
                ]
            })
        
        # Analyze checkout completion
        if drop_off_rates['checkout_to_confirmation'] < 0.5:
            suggestions.append({
                'step': 'Checkout Completion',
                'issue': 'High abandonment during checkout',
                'suggestions': [
                    'Simplify the checkout form',
                    'Add more payment options',
                    'Implement guest checkout',
                    'Show progress indicator'
                ]
            })
        
        # Analyze form filling patterns
        if patterns['form_filling']['completion_rate'] < 0.4:
            suggestions.append({
                'step': 'Form Filling',
                'issue': 'Low form completion rate',
                'suggestions': [
                    'Reduce number of required fields',
                    'Implement auto-fill for returning customers',
                    'Add form validation with clear error messages'
                ]
            })
        
        return suggestions

def visualize_checkout_analysis(df, analysis_results):
    # Create a funnel visualization
    plt.figure(figsize=(10, 6))
    steps = ['Cart View', 'Checkout Initiated', 'Checkout Confirmed']
    values = [
        df['No_Cart_Viewed'].sum(),
        df['No_Checkout_Initiated'].sum(),
        df['No_Checkout_Confirmed'].sum()
    ]
    
    plt.barh(steps, values)
    plt.title('Checkout Funnel Analysis')
    plt.xlabel('Number of Users')
    
    # Add percentage labels
    for i, v in enumerate(values):
        if i > 0:
            percentage = (v / values[i-1]) * 100
            plt.text(v, i, f'{percentage:.1f}%', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Create a heatmap of checkout steps
    plt.figure(figsize=(10, 6))
    steps_data = pd.DataFrame({
        'Cart View': df['No_Cart_Viewed'],
        'Checkout Initiated': df['No_Checkout_Initiated'],
        'Checkout Confirmed': df['No_Checkout_Confirmed']
    })
    
    sns.heatmap(steps_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Checkout Steps Correlation')
    plt.tight_layout()
    plt.show()

def main():
    # Load the data
    df = pd.read_csv("cleaned_cart_abandonment_data.csv")
    
    # Initialize analyzer
    analyzer = CheckoutAnalyzer()
    
    # Perform analysis
    results = analyzer.analyze_checkout_process(df)
    
    # Print results
    print("\nCheckout Analysis Results:")
    print("\nDrop-off Rates:")
    for step, rate in results['drop_off_rates'].items():
        print(f"{step}: {rate:.2%}")
    
    print("\nOptimization Suggestions:")
    for suggestion in results['suggestions']:
        print(f"\nStep: {suggestion['step']}")
        print(f"Issue: {suggestion['issue']}")
        print("Suggested Improvements:")
        for s in suggestion['suggestions']:
            print(f"- {s}")
    
    # Visualize results
    visualize_checkout_analysis(df, results)

if __name__ == "__main__":
    main() 