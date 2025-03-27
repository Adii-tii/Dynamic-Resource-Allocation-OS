import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import random

seed=random.seed(1024)
print(random.randint(7,12))

def train_test_pipeline(input_file='/kaggle/working/labelled_data.csv'):
    data = pd.read_csv(input_file)
    
    data['Original_Process_Name'] = data['Process_Name']
    # Encode Process_Name
    label_encoder = LabelEncoder()
    data['Process_Name'] = label_encoder.fit_transform(data['Process_Name'])

    high_priority_processes = ['chrome.exe', 'firefox.exe', 'code.exe', 'Adobe Illustrator']
    data['User_Priority'] = 0  # Default
    data.loc[data['Original_Process_Name'].isin(high_priority_processes), 'User_Priority'] = 10  # High

    features = [
        'Total_CPU_%', 'Total_RAM_%', 'Free_RAM_%', 'CPU_Usage_%', 
        'CPU_Time_s', 'Context_Switches', 'Memory_Usage_%', 'VMS_MB', 
        'Thread_Count', 'User_Priority', 'Process_Name'
    ]
    labels = [
        'Throttle', 'Prioritize', 'Reduce_Switching', 'Balance_Load', 
        'Boost_Parallel', 'Handle_Burst', 'Swap_Out', 'Optimize_Memory', 
        'Increase_VMS', 'Reduce_VMS', 'Prioritize_RAM'
    ]

    X = data[features]
    y = data[labels]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model = MultiOutputClassifier(rf, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = {}
    for i, label in enumerate(labels):
        report[label] = classification_report(y_test[label], y_pred[:, i], zero_division=0, output_dict=True)

    print("\nClassification Report:")
    for label, metrics in report.items():
        print(f"\n{label}:")
        print(pd.DataFrame(metrics).transpose())

    return model, scaler, label_encoder, report

# Run the pipeline
model, scaler, label_encoder, report = train_test_pipeline(input_file='/kaggle/working/labelled_data.csv')
