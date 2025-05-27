import pandas as pd
import joblib

# Load the Excel file with ticket data
df = pd.read_excel('data/raw/ai_dev_assignment_tickets_complex_1000.xls')  # Make sure this file is in the same directory or provide full path

# Extract the unique product names
product_list = df['product'].dropna().unique().tolist()

# Save the list to a .pkl file for use in your pipeline
joblib.dump(product_list, 'models/product_list.pkl')

print("✅ Product list saved to models/product_list.pkl")
