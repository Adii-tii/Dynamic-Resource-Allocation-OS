import streamlit as st
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from collections import deque
import random
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Advanced System Resource Monitor",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {
        background-color: #f0f5ff;
    }
    .strategy-tag {
        background-color: #4CAF50;
        color: white;
        padding: 2px 6px;
        margin: 2px;
        border-radius: 5px;
        font-size: 0.85em;
    }
    .strategy-container {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e6f0ff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4169E1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

if 'time_series_data' not in st.session_state:
    st.session_state.time_series_data = deque(maxlen=100)
    st.session_state.start_time = time.time()
    st.session_state.process_data = []

features = [
    'Total_CPU_%', 'Total_RAM_%', 'Free_RAM_%', 'CPU_Usage_Per_Process_%', 
    'CPU_Time_s', 'Context_Switches', 'Memory_Usage_%', 'VMS_MB', 
    'Thread_Count', 'User_Priority', 'Process_Name'
]

labels = [
    'Throttle', 'Prioritize', 'Reduce_Switching', 'Balance_Load', 
    'Boost_Parallel', 'Handle_Burst', 'Swap_Out', 'Optimize_Memory', 
    'Increase_VMS', 'Reduce_VMS', 'Prioritize_RAM'
]

strategy_colors = {
    'Throttle': '#FF5733',
    'Prioritize': '#33FF57',
    'Reduce_Switching': '#3357FF',
    'Balance_Load': '#F3FF33',
    'Boost_Parallel': '#33FFF3',
    'Handle_Burst': '#F333FF',
    'Swap_Out': '#FF33A1',
    'Optimize_Memory': '#A1FF33',
    'Increase_VMS': '#33A1FF',
    'Reduce_VMS': '#FFA133',
    'Prioritize_RAM': '#A133FF'
}

st.title("Advanced System Resource Monitor")
st.markdown("Real-time monitoring and optimization strategies for system resources")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Dashboard", 
    "🔍 Process Analysis", 
    "🎯 Action Prediction", 
    "🧪 Simulation", 
    "📊 Visualization"
])

def get_process_data():
    process_data = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads']):
            try:
                info = proc.info
                try:
                    info['cpu_time'] = sum(proc.cpu_times()[:2]) if hasattr(proc, 'cpu_times') else 0
                except:
                    info['cpu_time'] = 0
                try:
                    info['vms'] = proc.memory_info().vms / (1024 * 1024) if hasattr(proc, 'memory_info') else 0
                except:
                    info['vms'] = 0
                process_data.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        st.error(f"Error accessing process data: {str(e)}")
    return process_data

def predict_strategies(process_data):
    strategies = []
    total_cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    total_ram = memory.percent
    free_ram = 100 - total_ram
    cpu_percent = process_data.get('cpu_percent', 0)
    memory_percent = process_data.get('memory_percent', 0)
    thread_count = process_data.get('num_threads', 0)
    vms_mb = process_data.get('vms', 0)
    cpu_time = process_data.get('cpu_time', 0)
    process_name = process_data.get('name', '')
    if cpu_percent < 1.0 and memory_percent < 1.0:
        return []
    if total_cpu > 85 and cpu_percent > 20:
        strategies.append('Throttle')
    if thread_count > 50 and cpu_percent > 5:
        strategies.append('Reduce_Switching')
    important_processes = ['chrome.exe', 'firefox.exe', 'code.exe', 'python.exe', 'systemd']
    if any(proc in str(process_name).lower() for proc in important_processes) and cpu_percent > 10:
        strategies.append('Prioritize')
    if cpu_percent > 30 and thread_count > 30:
        strategies.append('Boost_Parallel')
    if free_ram < 15 and memory_percent > 5:
        strategies.append('Swap_Out')
    if memory_percent > 30:
        strategies.append('Optimize_Memory')
    if vms_mb > 8000 and memory_percent > 10:
        strategies.append('Increase_VMS')
    elif vms_mb < 2000 and memory_percent > 30:
        strategies.append('Reduce_VMS')
    if total_ram > 75 and memory_percent > 10:
        strategies.append('Prioritize_RAM')
    context_switches = random.randint(1000, 200000)
    if context_switches > 100000 and cpu_percent > 10:
        strategies.append('Handle_Burst')
    if 40 < total_cpu < 70 and thread_count > 20 and cpu_percent > 5:
        strategies.append('Balance_Load')
    return list(set(strategies))

def update_time_series():
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    current_time = time.time() - st.session_state.start_time
    st.session_state.time_series_data.append({
        'time': current_time,
        'cpu': cpu_usage,
        'memory': memory_usage,
        'free_memory': 100 - memory_usage,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    st.session_state.process_data = get_process_data()

with tab1:
    update_time_series()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cpu_usage = psutil.cpu_percent()
        delta = None
        if cpu_usage > 50:
            delta = f"{cpu_usage - 50:.1f}%"
        st.metric("CPU Usage", f"{cpu_usage}%", delta)
    with col2:
        memory = psutil.virtual_memory()
        delta = None
        if memory.percent > 50:
            delta = f"{memory.percent - 50:.1f}%"
        st.metric("Memory Usage", f"{memory.percent}%", delta)
    with col3:
        disk = psutil.disk_usage('/')
        delta = None
        if disk.percent > 50:
            delta = f"{disk.percent - 50:.1f}%"
        st.metric("Disk Usage", f"{disk.percent}%", delta)
    with col4:
        process_count = len(list(psutil.process_iter()))
        st.metric("Running Processes", process_count)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CPU & Memory Usage Over Time")
        if st.session_state.time_series_data:
            df = pd.DataFrame(list(st.session_state.time_series_data))
            fig = px.line(df, x='time', y=['cpu', 'memory'], 
                      labels={"value": "Usage (%)", "time": "Time (seconds)", "variable": "Resource"},
                      color_discrete_map={"cpu": "#2A9D8F", "memory": "#E76F51"})
            fig.update_layout(
                height=400,
                legend_title_text='',
                xaxis_title="Time (seconds)",
                yaxis_title="Usage (%)",
                yaxis_range=[0, 100],
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting data... Please wait.")
    with col2:
        st.subheader("Process CPU Usage")
        if st.session_state.process_data:
            process_df = pd.DataFrame(st.session_state.process_data)
            process_df = process_df[process_df['cpu_percent'] > 0.1]
            process_df = process_df.sort_values('cpu_percent', ascending=False).head(5)
            if not process_df.empty:
                fig = px.bar(process_df, x='name', y='cpu_percent', 
                          labels={"cpu_percent": "CPU Usage (%)", "name": "Process Name"},
                          color='cpu_percent', color_continuous_scale=px.colors.sequential.Viridis)
                fig.update_layout(
                    height=400,
                    xaxis_title="Process",
                    yaxis_title="CPU Usage (%)",
                    yaxis_range=[0, max(100, process_df['cpu_percent'].max() * 1.1)]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No active processes detected.")
        else:
            st.info("No process data available.")
    st.subheader("Top CPU-Consuming Processes")
    if st.session_state.process_data:
        process_df = pd.DataFrame(st.session_state.process_data)
        process_df = process_df[process_df['cpu_percent'] > 0.1]
        if not process_df.empty:
            process_df = process_df.sort_values('cpu_percent', ascending=False).head(10)
            process_df = process_df[['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads']]
            process_df.columns = ['PID', 'Process Name', 'CPU %', 'Memory %', 'Threads']
            st.dataframe(process_df, use_container_width=True)
        else:
            st.info("No active processes detected.")
    else:
        st.info("No process data available.")

with tab2:
    st.subheader("Process Analysis with Strategy Prediction")
    process_data = get_process_data()
    if process_data:
        process_df = pd.DataFrame(process_data)
        process_df = process_df[(process_df['cpu_percent'] > 0.1) | (process_df['memory_percent'] > 0.1)]
        if not process_df.empty:
            process_df['strategies'] = process_df.apply(lambda row: predict_strategies(row), axis=1)
            st.subheader("Active Processes with Recommended Strategies")
            process_df = process_df.sort_values('cpu_percent', ascending=False)
            display_df = process_df[['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads', 'vms', 'strategies']]
            display_df.columns = ['PID', 'Process Name', 'CPU %', 'Memory %', 'Threads', 'VMS (MB)', 'Strategies']
            def format_strategies(strategies):
                if not strategies:
                    return '<span class="strategy-tag" style="background-color: #808080">No optimization needed</span>'
                tags = ""
                for strategy in strategies:
                    tags += f'<span class="strategy-tag" style="background-color: {strategy_colors.get(strategy, "#4CAF50")}">{strategy}</span> '
                return f'<div class="strategy-container">{tags}</div>'
            display_df['Strategies'] = display_df['Strategies'].apply(
                lambda x: format_strategies(x)
            )
            st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.subheader("Process Detail View")
            process_names = process_df['name'].unique().tolist()
            if process_names:
                selected_process = st.selectbox("Select a process for detailed analysis", process_names)
                if selected_process:
                    process_detail = process_df[process_df['name'] == selected_process].iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("CPU Usage", f"{process_detail['cpu_percent']:.2f}%")
                        st.metric("Memory Usage", f"{process_detail['memory_percent']:.2f}%")
                    with col2:
                        st.metric("PID", process_detail['pid'])
                        st.metric("Thread Count", process_detail['num_threads'])
                    with col3:
                        st.metric("Virtual Memory", f"{process_detail['vms']:.2f} MB")
                        st.metric("Strategy Count", len(process_detail['strategies']))
                    st.subheader("Recommended Strategies")
                    if not process_detail['strategies']:
                        st.success("No optimization needed for this process - it's using resources efficiently.")
                    else:
                        for strategy in process_detail['strategies']:
                            with st.expander(f"{strategy}"):
                                if strategy == 'Throttle':
                                    st.write("Reduce CPU allocation to prevent system slowdown")
                                    st.write("**Why?** This process is using a high percentage of CPU, which could affect system responsiveness.")
                                elif strategy == 'Prioritize':
                                    st.write("Increase process priority to ensure responsive behavior")
                                    st.write("**Why?** This is an important application that should receive adequate resources.")
                                elif strategy == 'Reduce_Switching':
                                    st.write("Minimize context switching to improve performance")
                                    st.write("**Why?** This process has many threads which could cause excessive context switching.")
                                elif strategy == 'Balance_Load':
                                    st.write("Distribute CPU load more evenly across cores")
                                    st.write("**Why?** The process could benefit from better CPU core utilization.")
                                elif strategy == 'Boost_Parallel':
                                    st.write("Optimize thread utilization for parallel processing")
                                    st.write("**Why?** This multi-threaded process could benefit from optimized thread scheduling.")
                                elif strategy == 'Handle_Burst':
                                    st.write("Optimize for bursty workload patterns")
                                    st.write("**Why?** The process shows signs of intermittent high resource usage.")
                                elif strategy == 'Swap_Out':
                                    st.write("Move process memory to swap space to free up RAM")
                                    st.write("**Why?** System memory is running low and this process uses significant memory.")
                                elif strategy == 'Optimize_Memory':
                                    st.write("Reduce memory footprint through optimization")
                                    st.write("**Why?** This process is using more memory than expected for its functionality.")
                                elif strategy == 'Increase_VMS':
                                    st.write("Allocate more virtual memory space")
                                    st.write("**Why?** The process is approaching its virtual memory limits.")
                                elif strategy == 'Reduce_VMS':
                                    st.write("Decrease virtual memory allocation")
                                    st.write("**Why?** The process is using excessive virtual memory relative to its actual needs.")
                                elif strategy == 'Prioritize_RAM':
                                    st.write("Keep process data in physical RAM for better performance")
                                    st.write("**Why?** This is a performance-critical process that should avoid page faults.")
            else:
                st.info("No active processes with significant resource usage detected.")
        else:
            st.info("No processes with significant resource usage detected.")
    else:
        st.info("No process data available for analysis.")

with tab3:
    st.subheader("Action Prediction")
    st.write("Predict optimization strategies based on resource parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System Parameters")
        total_cpu = st.slider("Total CPU Usage (%)", 0, 100, 50)
        total_ram = st.slider("Total RAM Usage (%)", 0, 100, 60)
        free_ram = st.slider("Free RAM (%)", 0, 100, 40)
        context_switches = st.slider("Context Switches", 0, 500000, 50000)
    with col2:
        st.subheader("Process Parameters")
        cpu_per_process = st.slider("CPU Usage Per Process (%)", 0, 100, 30)
        cpu_time = st.slider("CPU Time (s)", 0, 1000, 100)
        memory_usage = st.slider("Memory Usage (%)", 0, 100, 40)
        vms_mb = st.slider("Virtual Memory Size (MB)", 0, 10000, 2000)
        thread_count = st.slider("Thread Count", 0, 100, 20)
        user_priority = st.slider("User Priority", 0, 10, 5)
        process_name = st.selectbox("Process Name", ["chrome.exe", "firefox.exe", "code.exe", "python.exe", "systemd", "other"])
    if st.button("Predict Actions"):
        strategies = []
        is_high_load = total_cpu > 70 or total_ram > 80 or free_ram < 20
        is_significant_process = cpu_per_process > 5 or memory_usage > 5
        if is_high_load or is_significant_process:
            if total_cpu > 80 and cpu_per_process > 10:
                strategies.append('Throttle')
            if thread_count > 50 and cpu_per_process > 5:
                strategies.append('Reduce_Switching')
            if user_priority > 7 or process_name in ['chrome.exe', 'firefox.exe', 'code.exe'] and cpu_per_process > 5:
                strategies.append('Prioritize')
            if cpu_per_process > 30 and thread_count > 30:
                strategies.append('Boost_Parallel')
            if free_ram < 15 and memory_usage > 5:
                strategies.append('Swap_Out')
            if memory_usage > 30:
                strategies.append('Optimize_Memory')
            if vms_mb > 8000 and memory_usage > 10:
                strategies.append('Increase_VMS')
            elif vms_mb < 2000 and memory_usage > 30:
                strategies.append('Reduce_VMS')
            if total_ram > 75 and memory_usage > 10:
                strategies.append('Prioritize_RAM')
            if context_switches > 100000 and cpu_per_process > 10:
                strategies.append('Handle_Burst')
            if 40 < total_cpu < 70 and thread_count > 20 and cpu_per_process > 5:
                strategies.append('Balance_Load')
        st.subheader("Prediction Results")
        if not strategies:
            st.success("No optimization strategies needed at this time. The system and process resources are within normal parameters.")
        else:
            st.write(f"**{len(strategies)} strategies recommended:**")
            for strategy in strategies:
                with st.expander(f"{strategy}", expanded=True):
                    if strategy == 'Throttle':
                        st.write("Reduce CPU allocation to prevent system slowdown")
                        st.progress(min(total_cpu, 100))
                        st.write(f"**Why?** CPU usage is high ({total_cpu}%), and this process is using {cpu_per_process}% of CPU.")
                    elif strategy == 'Prioritize':
                        st.write("Increase process priority to ensure responsive behavior")
                        st.progress(min(user_priority * 10, 100))
                        st.write(f"**Why?** This is a priority application that should maintain responsive behavior.")
                    elif strategy == 'Reduce_Switching':
                        st.write("Minimize context switching to improve performance")
                        st.progress(min(thread_count, 100))
                        st.write(f"**Why?** High thread count ({thread_count}) may cause excessive context switching.")
                    elif strategy == 'Balance_Load':
                        st.write("Distribute CPU load more evenly across cores")
                        st.progress(min(total_cpu, 100))
                        st.write(f"**Why?** Moderate-high CPU usage ({total_cpu}%) could benefit from balanced core utilization.")
                    elif strategy == 'Boost_Parallel':
                        st.write("Optimize thread utilization for parallel processing")
                        st.progress(min(thread_count, 100))
                        st.write(f"**Why?** Process has many threads ({thread_count}) that could benefit from parallel optimization.")
                    elif strategy == 'Handle_Burst':
                        st.write("Optimize for bursty workload patterns")
                        st.progress(min(context_switches / 5000, 100))
                        st.write(f"**Why?** High context switch count ({context_switches}) indicates bursty execution patterns.")
                    elif strategy == 'Swap_Out':
                        st.write("Move process memory to swap space to free up RAM")
                        st.progress(min((100 - free_ram), 100))
                        st.write(f"**Why?** Low free RAM ({free_ram}%) indicates need for memory management.")
                    elif strategy == 'Optimize_Memory':
                        st.write("Reduce memory footprint through optimization")
                        st.progress(min(memory_usage, 100))
                        st.write(f"**Why?** Process is using significant memory ({memory_usage}%) that could be optimized.")
                    elif strategy == 'Increase_VMS':
                        st.write("Allocate more virtual memory space")
                        st.progress(min(vms_mb / 100, 100))
                        st.write(f"**Why?** Large virtual memory footprint ({vms_mb} MB) may require additional allocation.")
                    elif strategy == 'Reduce_VMS':
                        st.write("Decrease virtual memory allocation")
                        st.progress(min((10000 - vms_mb) / 100, 100))
                        st.write(f"**Why?** Virtual memory usage is inefficient for actual memory needs.")
                    elif strategy == 'Prioritize_RAM':
                        st.write("Keep process data in physical RAM for better performance")
                        st.progress(min(total_ram, 100))
                        st.write(f"**Why?** High system RAM usage ({total_ram}%) means critical processes should be kept in physical memory.")

with tab4:
    st.subheader("Resource Allocation Simulation")
    st.write("Configure the simulation environment")
    col1, col2 = st.columns(2)
    with col1:
        sim_duration = st.slider("Simulation Duration (s)", 10, 120, 30)
        environment = st.selectbox(
            "Simulation Environment", 
            ["High CPU Load", "High Memory Load", "Balanced Load", "Low Load", "Random Pattern"]
        )
    with col2:
        process_count = st.slider("Number of Processes", 3, 10, 5)
        update_interval = st.slider("Update Interval (ms)", 500, 5000, 1000)
    start_simulation = st.button("Start Simulation")
    if start_simulation:
        simulation_container = st.container()
        progress_bar = st.progress(0)
        sim_data = []
        processes = [f"Process-{i}" for i in range(1, process_count+1)]
        env_patterns = {
            "High CPU Load": {'cpu_base': 80, 'memory_base': 40, 'threshold': 'high'},
            "High Memory Load": {'cpu_base': 30, 'memory_base': 80, 'threshold': 'high'},
            "Balanced Load": {'cpu_base': 60, 'memory_base': 60, 'threshold': 'medium'},
            "Low Load": {'cpu_base': 20, 'memory_base': 30, 'threshold': 'low'},
            "Random Pattern": {'cpu_base': 50, 'memory_base': 50, 'threshold': 'variable'}
        }
        start_time = time.time()
        iterations = int(sim_duration * 1000 / update_interval)
        for i in range(iterations):
            progress = (i + 1) / iterations
            progress_bar.progress(progress)
            env = env_patterns[environment]
            cpu_base = env['cpu_base']
            memory_base = env['memory_base']
            threshold = env['threshold']
            if environment == "Random Pattern":
                cpu_base = random.uniform(20, 80)
                memory_base = random.uniform(20, 80)
                if cpu_base > 70 or memory_base > 70:
                    threshold = 'high'
                elif cpu_base < 30 and memory_base < 30:
                    threshold = 'low'
                else:
                    threshold = 'medium'
            cpu_base += random.uniform(-10, 10)
            memory_base += random.uniform(-10, 10)
            cpu_base = max(0, min(100, cpu_base))
            memory_base = max(0, min(100, memory_base))
            timestamp = time.time() - start_time
            process_data = []
            for proc in processes:
                cpu_percent = cpu_base * random.uniform(0.3, 1.5)
                memory_percent = memory_base * random.uniform(0.3, 1.5)
                if threshold == 'low':
                    if random.random() < 0.7:
                        cpu_percent *= 0.3
                        memory_percent *= 0.3
                cpu_percent = max(0, min(100, cpu_percent))
                memory_percent = max(0, min(100, memory_percent))
                if cpu_percent > 60:
                    thread_count = random.randint(20, 100)
                elif cpu_percent > 30:
                    thread_count = random.randint(10, 50)
                else:
                    thread_count = random.randint(1, 20)
                vms = memory_percent * random.uniform(100, 200)
                process_entry = {
                    'name': proc,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'num_threads': thread_count,
                    'vms': vms,
                    'timestamp': timestamp
                }
                if threshold == 'low' and cpu_percent < 10 and memory_percent < 10:
                    process_entry['strategies'] = []
                else:
                    strategies = []
                    if cpu_percent > 70:
                        strategies.append('Throttle')
                    if thread_count > 50 and cpu_percent > 5:
                        strategies.append('Reduce_Switching')
                    if cpu_percent > 30 and thread_count > 30:
                        strategies.append('Boost_Parallel')
                    if memory_base > 75 and memory_percent > 15:
                        strategies.append('Swap_Out')
                    if memory_percent > 60:
                        strategies.append('Optimize_Memory')
                    if vms > 5000 and memory_percent > 20:
                        strategies.append('Increase_VMS')
                    elif vms < 500 and memory_percent > 40:
                        strategies.append('Reduce_VMS')
                    if memory_base > 75 and memory_percent > 15:
                        strategies.append('Prioritize_RAM')
                    if 40 < cpu_base < 70 and thread_count > 20 and cpu_percent > 5:
                        strategies.append('Balance_Load')
                    process_entry['strategies'] = list(set(strategies))
                process_data.append(process_entry)
            system_entry = {
                'total_cpu': cpu_base,
                'total_memory': memory_base,
                'free_memory': 100 - memory_base,
                'timestamp': timestamp
            }
            sim_data.append({
                'system': system_entry,
                'processes': process_data
            })
            with simulation_container:
                st.subheader(f"Simulation Step {i+1}/{iterations}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("System CPU", f"{cpu_base:.1f}%")
                    st.metric("System Memory", f"{memory_base:.1f}%")
                process_df = pd.DataFrame(process_data)
                process_df = process_df.sort_values('cpu_percent', ascending=False)
                with col2:
                    st.write("Top CPU Processes:")
                    top_procs = process_df[['name', 'cpu_percent']].head(3)
                    top_procs.columns = ['Process', 'CPU %']
                    st.dataframe(top_procs)
                cpu_df = process_df.sort_values('cpu_percent', ascending=False).head(5)
                fig = px.bar(cpu_df, x='name', y='cpu_percent',
                          labels={"cpu_percent": "CPU Usage (%)", "name": "Process Name"},
                          title="Top CPU Consumers",
                          color='cpu_percent', color_continuous_scale=px.colors.sequential.Viridis)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                process_df['strategy_count'] = process_df['strategies'].apply(len)
                if process_df['strategy_count'].sum() > 0:
                    st.subheader("Recommended Optimization Actions")
                    critical_proc = process_df.sort_values('strategy_count', ascending=False).iloc[0]
                    if critical_proc['strategy_count'] > 0:
                        st.write(f"**Critical Process:** {critical_proc['name']}")
                        st.write(f"**Recommended Actions:** {', '.join(critical_proc['strategies'])}")
                else:
                    st.success("No optimization needed at this step - all processes running efficiently")
            if i == iterations - 1:
                st.session_state.simulation_results = sim_data
            time.sleep(update_interval / 1000)
        st.success(f"Simulation completed with {len(sim_data)} data points")
        if st.button("Download Simulation Results"):
            all_process_data = []
            for step in sim_data:
                system = step['system']
                for proc in step['processes']:
                    entry = {
                        'timestamp': system['timestamp'],
                        'total_cpu': system['total_cpu'],
                        'total_memory': system['total_memory'],
                        'free_memory': system['free_memory'],
                        'process_name': proc['name'],
                        'cpu_percent': proc['cpu_percent'],
                        'memory_percent': proc['memory_percent'],
                        'thread_count': proc['num_threads'],
                        'vms': proc['vms'],
                        'strategy_count': len(proc['strategies']),
                        'strategies': ','.join(proc['strategies'])
                    }
                    all_process_data.append(entry)
            df = pd.DataFrame(all_process_data)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="simulation_results.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

with tab5:
    st.subheader("Resource Usage Visualization")
    has_real_data = len(st.session_state.time_series_data) > 0
    has_sim_data = 'simulation_results' in st.session_state
    if not has_real_data and not has_sim_data:
        st.info("Please run a simulation or collect real-time data first.")
    else:
        data_source = st.radio(
            "Select Data Source",
            ["Real-time Data" if has_real_data else "No Real-time Data Available", 
             "Simulation Data" if has_sim_data else "No Simulation Data Available"]
        )
        if data_source.startswith("Real-time") and has_real_data:
            df = pd.DataFrame(list(st.session_state.time_series_data))
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Line Chart", "Area Chart", "Bar Chart", "Scatter Plot", "Heatmap"]
            )
            st.subheader(f"System Resource Usage - {chart_type}")
            if chart_type == "Line Chart":
                fig = px.line(df, x='time', y=['cpu', 'memory', 'free_memory'],
                           labels={"value": "Usage (%)", "time": "Time (seconds)", "variable": "Resource"},
                           color_discrete_map={"cpu": "#2A9D8F", "memory": "#E76F51", "free_memory": "#66a1e5"})
                fig.update_layout(
                    height=500,
                    legend_title_text='',
                    xaxis_title="Time (seconds)",
                    yaxis_title="Usage (%)",
                    yaxis_range=[0, 100],
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Area Chart":
                fig = px.area(df, x='time', y=['cpu', 'memory'],
                           labels={"value": "Usage (%)", "time": "Time (seconds)", "variable": "Resource"},
                           color_discrete_map={"cpu": "#2A9D8F", "memory": "#E76F51"})
                fig.update_layout(
                    height=500,
                    legend_title_text='',
                    xaxis_title="Time (seconds)",
                    yaxis_title="Usage (%)",
                    yaxis_range=[0, 100],
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Bar Chart":
                avg_data = {
                    'Resource': ['CPU', 'Memory', 'Free Memory'],
                    'Average Usage (%)': [
                        df['cpu'].mean(),
                        df['memory'].mean(),
                        df['free_memory'].mean()
                    ]
                }
                avg_df = pd.DataFrame(avg_data)
                fig = px.bar(avg_df, x='Resource', y='Average Usage (%)',
                          color='Resource',
                          color_discrete_map={"CPU": "#2A9D8F", "Memory": "#E76F51", "Free Memory": "#66a1e5"})
                fig.update_layout(
                    height=500,
                    yaxis_range=[0, 100]
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df, x='cpu', y='memory', color='free_memory',
                              labels={"cpu": "CPU Usage (%)", "memory": "Memory Usage (%)", "free_memory": "Free Memory (%)"},
                              color_continuous_scale=px.colors.sequential.Viridis)
                fig.update_layout(
                    height=500,
                    xaxis_title="CPU Usage (%)",
                    yaxis_title="Memory Usage (%)",
                    xaxis_range=[0, 100],
                    yaxis_range=[0, 100]
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Heatmap":
                corr_df = df[['cpu', 'memory', 'free_memory']].corr()
                fig = px.imshow(corr_df, text_auto=True, aspect="equal",
                             color_continuous_scale=px.colors.diverging.RdBu_r)
                fig.update_layout(
                    height=500,
                    title="Resource Usage Correlation"
                )
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("Resource Usage Insights")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg CPU Usage", f"{df['cpu'].mean():.1f}%")
                st.metric("Max CPU Usage", f"{df['cpu'].max():.1f}%")
            with col2:
                st.metric("Avg Memory Usage", f"{df['memory'].mean():.1f}%")
                st.metric("Max Memory Usage", f"{df['memory'].max():.1f}%")
            with col3:
                st.metric("Avg Free Memory", f"{df['free_memory'].mean():.1f}%")
                st.metric("Min Free Memory", f"{df['free_memory'].min():.1f}%")
            st.subheader("System Status Assessment")
            cpu_mean = df['cpu'].mean()
            memory_mean = df['memory'].mean()
            if cpu_mean > 80 or memory_mean > 80:
                st.error("⚠️ System is under heavy load - optimization recommended")
            elif cpu_mean > 60 or memory_mean > 60:
                st.warning("⚠️ System is under moderate load - monitor closely")
            else:
                st.success("✅ System is running efficiently")
        elif data_source.startswith("Simulation") and has_sim_data:
            sim_data = st.session_state.simulation_results
            system_metrics = []
            for step in sim_data:
                system_metrics.append(step['system'])
            system_df = pd.DataFrame(system_metrics)
            sim_chart_type = st.selectbox(
                "Select Simulation Chart Type",
                ["System Overview", "Process Analysis", "Strategy Distribution", "Resource Usage Patterns"]
            )
            if sim_chart_type == "System Overview":
                st.subheader("System Resource Usage During Simulation")
                fig = px.line(system_df, x='timestamp', y=['total_cpu', 'total_memory', 'free_memory'],
                           labels={"value": "Usage (%)", "timestamp": "Time (seconds)", "variable": "Resource"},
                           color_discrete_map={"total_cpu": "#2A9D8F", "total_memory": "#E76F51", "free_memory": "#66a1e5"})
                fig.update_layout(
                    height=500,
                    legend_title_text='',
                    xaxis_title="Simulation Time (seconds)",
                    yaxis_title="Usage (%)",
                    yaxis_range=[0, 100],
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            elif sim_chart_type == "Process Analysis":
                st.subheader("Process Resource Usage Analysis")
                all_processes = []
                for step in sim_data:
                    for proc in step['processes']:
                        proc['system_cpu'] = step['system']['total_cpu']
                        proc['system_memory'] = step['system']['total_memory']
                        all_processes.append(proc)
                process_df = pd.DataFrame(all_processes)
                process_names = process_df['name'].unique().tolist()
                selected_process = st.selectbox("Select Process", process_names)
                if selected_process:
                    proc_data = process_df[process_df['name'] == selected_process]
                    fig = px.line(proc_data, x='timestamp', y=['cpu_percent', 'memory_percent'],
                               labels={"value": "Usage (%)", "timestamp": "Time (seconds)", "variable": "Resource"},
                               color_discrete_map={"cpu_percent": "#2A9D8F", "memory_percent": "#E76F51"})
                    fig.update_layout(
                        height=500,
                        legend_title_text='',
                        xaxis_title="Simulation Time (seconds)",
                        yaxis_title="Usage (%)",
                        yaxis_range=[0, 100],
                        hovermode="x unified",
                        title=f"Resource Usage for {selected_process}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Thread Count Analysis")
                    fig = px.line(proc_data, x='timestamp', y='num_threads',
                               labels={"num_threads": "Thread Count", "timestamp": "Time (seconds)"},
                               color_discrete_sequence=["#9D2A8F"])
                    fig.update_layout(
                        height=300,
                        xaxis_title="Simulation Time (seconds)",
                        yaxis_title="Thread Count",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            elif sim_chart_type == "Strategy Distribution":
                st.subheader("Optimization Strategy Distribution")
                all_strategies = []
                for step in sim_data:
                    for proc in step['processes']:
                        for strategy in proc['strategies']:
                            all_strategies.append({
                                'timestamp': proc['timestamp'],
                                'process': proc['name'],
                                'strategy': strategy,
                                'cpu': proc['cpu_percent'],
                                'memory': proc['memory_percent']
                            })
                if all_strategies:
                    strategy_df = pd.DataFrame(all_strategies)
                    strategy_counts = strategy_df['strategy'].value_counts().reset_index()
                    strategy_counts.columns = ['Strategy', 'Count']
                    fig = px.bar(strategy_counts, x='Strategy', y='Count',
                              color='Strategy', color_discrete_map=strategy_colors)
                    fig.update_layout(
                        height=400,
                        xaxis_title="Strategy",
                        yaxis_title="Occurrence Count",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Strategy Timing Analysis")
                    pivot_df = pd.pivot_table(
                        strategy_df, 
                        values='process', 
                        index='timestamp', 
                        columns='strategy', 
                        aggfunc='count',
                        fill_value=0
                    ).reset_index()
                    melted_df = pd.melt(pivot_df, id_vars=['timestamp'], var_name='strategy', value_name='count')
                    fig = px.line(melted_df, x='timestamp', y='count', color='strategy',
                               labels={"count": "Occurrence Count", "timestamp": "Time (seconds)", "strategy": "Strategy"},
                               color_discrete_map=strategy_colors)
                    fig.update_layout(
                        height=500,
                        legend_title_text='',
                        xaxis_title="Simulation Time (seconds)",
                        yaxis_title="Strategy Count",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No optimization strategies were triggered during the simulation.")
            elif sim_chart_type == "Resource Usage Patterns":
                st.subheader("Resource Usage Patterns")
                all_process_data = []
                for step in sim_data:
                    for proc in step['processes']:
                        proc['system_cpu'] = step['system']['total_cpu']
                        proc['system_memory'] = step['system']['total_memory']
                        proc['strategy_count'] = len(proc['strategies'])
                        all_process_data.append(proc)
                pattern_df = pd.DataFrame(all_process_data)
                fig = px.scatter(pattern_df, x='cpu_percent', y='memory_percent', 
                              color='strategy_count', size='num_threads',
                              hover_data=['name', 'timestamp'],
                              labels={"cpu_percent": "CPU Usage (%)", "memory_percent": "Memory Usage (%)",
                                      "strategy_count": "Strategy Count", "num_threads": "Thread Count"},
                              color_continuous_scale=px.colors.sequential.Viridis)
                fig.update_layout(
                    height=500,
                    xaxis_title="CPU Usage (%)",
                    yaxis_title="Memory Usage (%)",
                    xaxis_range=[0, 100],
                    yaxis_range=[0, 100]
                )
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Process Efficiency Analysis")
                pattern_df['efficiency_score'] = (pattern_df['cpu_percent'] + pattern_df['memory_percent']) / pattern_df['num_threads']
                efficiency_df = pattern_df.groupby('name').agg({
                    'efficiency_score': 'mean',
                    'cpu_percent': 'mean',
                    'memory_percent': 'mean',
                    'num_threads': 'mean',
                    'strategy_count': 'mean'
                }).reset_index()
                fig = px.bar(efficiency_df.sort_values('efficiency_score'), x='name', y='efficiency_score',
                          color='efficiency_score', color_continuous_scale=px.colors.sequential.Viridis_r,
                          labels={"efficiency_score": "Efficiency Score (lower is better)", "name": "Process Name"})
                fig.update_layout(
                    height=400,
                    xaxis_title="Process",
                    yaxis_title="Efficiency Score (lower is better)"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select a valid data source.")

st.markdown("---")
st.markdown("Advanced System Resource Monitor | Created with Streamlit")
st.markdown("Data is collected in real-time from your system.")

if st.button("Toggle Auto-Refresh"):
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    else:
        st.session_state.auto_refresh = not st.session_state.auto_refresh
    if st.session_state.auto_refresh:
        st.success("Auto-refresh enabled (5s intervals)")
    else:
        st.info("Auto-refresh disabled")

if 'auto_refresh' in st.session_state and st.session_state.auto_refresh:
    st.rerun()
