import psutil
import pandas as pd
import time
from datetime import datetime
import os

def collect_system_metrics():
    total_cpu_percent = psutil.cpu_percent(interval=1)  
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)  # Per-core usage
    total_ram = psutil.virtual_memory().percent 
    free_ram = 100 - total_ram 

    process_data = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads']):
        try:
            pid = proc.info['pid']
            name = proc.info['name']
            cpu_usage = proc.info['cpu_percent'] 
            memory_usage = proc.info['memory_percent']  
            thread_count = proc.info['num_threads']  

            with proc.oneshot():
                cpu_time = proc.cpu_times().user + proc.cpu_times().system 
                context_switches = proc.num_ctx_switches().voluntary + proc.num_ctx_switches().involuntary
                vms = proc.memory_info().vms / (1024 * 1024) 
                priority = proc.nice()  

            process_data.append({
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total_CPU_%': total_cpu_percent,
                'Total_RAM_%': total_ram,
                'Free_RAM_%': free_ram,
                'PID': pid,
                'Process_Name': name,
                'CPU_Usage_%': cpu_usage,
                'CPU_Affinity': len(proc.cpu_affinity()) if proc.cpu_affinity() else 8,  # Number of cores 
                'CPU_Time_s': cpu_time,
                'Context_Switches': context_switches,
                'Memory_Usage_%': memory_usage,
                'VMS_MB': vms,
                'User_Priority': priority,
                'Thread_Count': thread_count
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    return process_data

num_samples = 2000
all_data = []
sample_interval = 1  # Collect data every 1 second

for i in range(num_samples):
    try:
        data_point = collect_system_metrics()
        all_data.extend(data_point)
        print(f"Collected sample {i+1}/{num_samples}")
        time.sleep(sample_interval)  
    except Exception as e:
        print(f"Error during collection: {e}")
        continue

df = pd.DataFrame(all_data)

output_file = 'resource_data.csv'
df.to_csv(output_file, index=False)
print(f"Data collection completed. Saved to {output_file}")
print(f"Dataset shape: {df.shape}")
print(df.head())
