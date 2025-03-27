import gradio as gr
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from collections import deque
import time
import pickle
import os

# Load the pre-trained model, scaler, and label encoder
with open('/kaggle/input/model-assets-for-os/tensorflow2/default/1/model_assets.pkl', 'rb') as f:
    assets = pickle.load(f)

model = assets['model']
scaler = assets['scaler']
label_encoder = assets['label_encoder']

# Global variables for time series data
time_series_data = deque(maxlen=50)

# Define features and labels
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

# Make predictions function
def make_predictions(X_input, return_format='list'):
    if not isinstance(X_input, pd.DataFrame):
        X_input = pd.DataFrame([X_input], columns=features)
    
    X_input = X_input.copy()
    X_input['Process_Name'] = label_encoder.transform(X_input['Process_Name'])
    
    high_priority_processes = ['chrome.exe', 'firefox.exe', 'code.exe', 'Adobe Illustrator']
    X_input['User_Priority'] = 0
    X_input.loc[X_input['Process_Name'].isin(high_priority_processes), 'User_Priority'] = 10
    
    X_scaled = scaler.transform(X_input[features])
    y_pred = model.predict(X_scaled)
    
    predictions = []
    for i in range(len(X_scaled)):
        pred_dict = {label: int(pred) for label, pred in zip(labels, y_pred[i])}
        if return_format == 'dict':
            predictions.append(pred_dict)
        elif return_format == 'list':
            predicted_strategies = [label for label, pred in pred_dict.items() if pred == 1]
            predictions.append(predicted_strategies if predicted_strategies else ["No strategies recommended"])
        else:
            raise ValueError("return_format must be 'dict' or 'list'")
    
    return predictions[0] if len(predictions) == 1 else predictions

# Dashboard tab functions
def get_process_data():
    process_data = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        process_data.append(proc.info)
    df = pd.DataFrame(process_data)
    return df.sort_values(by='cpu_percent', ascending=False).head(20)

def plot_resource_usage():
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_usage = psutil.virtual_memory().percent
    current_time = time.time()
    time_series_data.append((current_time, cpu_usage, memory_usage))

    times, cpu_values, mem_values = zip(*time_series_data)
    times = [(t - times[0]) for t in times]

    plt.figure(figsize=(8, 4), facecolor='#f0f0f0')
    plt.plot(times, cpu_values, label='CPU Usage (%)', color='#2A9D8F', linewidth=2)
    plt.plot(times, mem_values, label='Memory Usage (%)', color='#E76F51', linewidth=2)

    plt.title('Real-Time Resource Monitoring', fontsize=12, pad=10)
    plt.xlabel('Time (seconds ago)', fontsize=10)
    plt.ylabel('Usage (%)', fontsize=10)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 100)
    plt.xlim(max(times), 0)
    plt.fill_between(times, cpu_values, alpha=0.1, color='#2A9D8F')
    plt.fill_between(times, mem_values, alpha=0.1, color='#E76F51')

    plt.tight_layout()
    plt.savefig('resource_plot.png')
    plt.close()
    return 'resource_plot.png'

# Action Prediction tab
def predict_action(cpu_usage, ram_usage, free_ram, cpu_usage_percent, cpu_time, context_switches, memory_usage, vms_mb, thread_count, user_priority, process_name):
    user_input = {
        'Total_CPU_%': cpu_usage,
        'Total_RAM_%': ram_usage,
        'Free_RAM_%': free_ram,
        'CPU_Usage_%': cpu_usage_percent,
        'CPU_Time_s': cpu_time,
        'Context_Switches': context_switches,
        'Memory_Usage_%': memory_usage,
        'VMS_MB': vms_mb,
        'Thread_Count': thread_count,
        'User_Priority': user_priority,
        'Process_Name': process_name
    }
    
    prediction = make_predictions(user_input, return_format='list')
    return f"Recommended Strategies: {', '.join(prediction)}"

# Dataset Analysis tab
def analyze_dataset(dataset_file):
    data = pd.read_csv(dataset_file.name)

    if not all(feature in data.columns for feature in features):
        missing = [f for f in features if f not in data.columns]
        return f"Error: Missing required features in dataset: {missing}", None

    predictions = make_predictions(data, return_format='dict')

    for label in labels:
        data[label] = [pred[label] for pred in predictions]

    output_file = "labeled_dataset.csv"
    data.to_csv(output_file, index=False)

    return data, output_file

# Model Settings tab
def model_settings(n_estimators, max_depth, min_samples_split):
    global model
    
    df = pd.read_csv("/kaggle/input/resource-data-updated/resource_data_augmented_new3.csv")
    data = df.copy()
    data['Process_Name'] = label_encoder.transform(data['Process_Name'])
    
    high_priority_processes = ['chrome.exe', 'firefox.exe', 'code.exe', 'Adobe Illustrator']
    data['User_Priority'] = 0
    data.loc[data['Process_Name'].isin(high_priority_processes), 'User_Priority'] = 10
    
    X = data[features]
    y = data[labels]
    X = scaler.transform(X)
    
    rf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth) if max_depth is not None else None,
        min_samples_split=int(min_samples_split),
        random_state=42
    )
    model = MultiOutputClassifier(rf, n_jobs=-1)
    model.fit(X, y)
    
    assets = {'model': model, 'scaler': scaler, 'label_encoder': label_encoder}
    with open('/kaggle/working/model_assets.pkl', 'wb') as f:
        pickle.dump(assets, f)
    
    return f"Model Updated: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}"

# System Insights tab
def system_insights():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    running_processes = len(list(psutil.process_iter()))

    insights = "System Insights:\n\n"
    if cpu_usage > 80:
        insights += f"- High CPU usage detected ({cpu_usage}%). Consider using 'Throttle' or 'Balance_Load' strategies.\n"
    else:
        insights += f"- CPU usage is within normal range ({cpu_usage}%).\n"
    if memory_usage > 80:
        insights += f"- High memory usage detected ({memory_usage}%). Consider using 'Swap_Out' or 'Optimize_Memory' strategies.\n"
    else:
        insights += f"- Memory usage is within normal range ({memory_usage}%).\n"
    if disk_usage > 90:
        insights += f"- High disk usage detected ({disk_usage}%). Free up disk space to improve performance.\n"
    else:
        insights += f"- Disk usage is within normal range ({disk_usage}%).\n"
    if running_processes > 200:
        insights += f"- High number of running processes ({running_processes}). Consider terminating unnecessary processes.\n"
    else:
        insights += f"- Number of running processes is reasonable ({running_processes}).\n"
    return insights

# Gradio Tabbed Interface
demo = gr.TabbedInterface(
    [
        gr.Interface(
            fn=lambda: (plot_resource_usage(), get_process_data()),
            inputs=[],
            outputs=[
                gr.Image(label="Resource Usage", show_label=True),
                gr.Dataframe(label="Active Processes", show_label=True)
            ],
            live=True,
            title="Dashboard",
            css="""
                .output-container {
                    display: flex;
                    flex-direction: row;
                    justify-content: space-between;
                    align-items: flex-start;
                    gap: 20px;
                }
                .output-image {
                    flex: 1;
                    max-width: 50%;
                }
                .output-dataframe {
                    flex: 1;
                    max-width: 50%;
                }
            """
        
        ),
        gr.Interface(
            fn=predict_action,
            inputs=[
                gr.Slider(0, 100, step=1, label="CPU Usage (%)"),
                gr.Slider(0, 100, step=1, label="RAM Usage (%)"),
                gr.Slider(0, 100, step=1, label="Free RAM (%)"),
                gr.Slider(0, 100, step=1, label="CPU Usage (%)"),
                gr.Slider(0, 1000, step=1, label="CPU Time (s)"),
                gr.Slider(0, 500000, step=1, label="Context Switches"),
                gr.Slider(0, 100, step=1, label="Memory Usage (%)"),
                gr.Slider(0, 10000, step=1, label="VMS (MB)"),
                gr.Slider(0, 100, step=1, label="Thread Count"),
                gr.Slider(0, 10, step=1, label="User Priority"),
                gr.Dropdown(label_encoder.classes_.tolist(), label="Process Name")
            ],
            outputs="text",
            title="Action Prediction"
        ),
        gr.Interface(
            fn=analyze_dataset,
            inputs=gr.File(label="Upload Dataset (.csv)"),
            outputs=[gr.Dataframe(label="Processed Data"), gr.File(label="Download Processed Dataset")],
            title="Dataset Analysis"
        ),
        gr.Interface(
            fn=model_settings,
            inputs=[
                gr.Number(value=100, label="n_estimators"),
                gr.Number(value=None, label="max_depth"),
                gr.Number(value=2, label="min_samples_split")
            ],
            outputs="text",
            title="Model Settings"
        ),
        gr.Interface(
            fn=system_insights,
            inputs=[],
            outputs="text",
            title="System Insights"
        )
    ],
    tab_names=["Dashboard", "Action Prediction", "Dataset Analysis", "Model Settings", "System Insights"]
)

if __name__ == "__main__":
    demo.launch()
