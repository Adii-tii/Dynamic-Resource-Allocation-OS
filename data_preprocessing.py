#data labelling
def label_data(input_file, output_file='resource_data_labeled.csv'):
    data = pd.read_csv(input_file)
    print("Data loaded. Shape:", data.shape)

    data['Throttle'] = 0
    data['Prioritize'] = 0
    data['Reduce_Switching'] = 0
    data['Balance_Load'] = 0
    data['Boost_Parallel'] = 0
    data['Handle_Burst'] = 0
    data['Swap_Out'] = 0
    data['Optimize_Memory'] = 0
    data['Increase_VMS'] = 0
    data['Reduce_VMS'] = 0
    data['Prioritize_RAM'] = 0

    # Apply labeling conditions for cpu
    data.loc[(data['Total_CPU_%'] > 75) & (data['CPU_Usage_%'] > 200), 'Throttle'] = 1
    data.loc[(data['CPU_Usage_%'] < 30) & (data['Total_CPU_%'] > 60), 'Prioritize'] = 1
    data.loc[(data['Context_Switches'] > 1000000) & (data['Total_CPU_%'] > 40), 'Reduce_Switching'] = 1
    underutilized = data.groupby('Timestamp').apply(lambda x: (x['CPU_Usage_%'] < 40).sum() > 2)
    underutilized_timestamps = underutilized[underutilized].index
    data.loc[(data['Timestamp'].isin(underutilized_timestamps)) & (data['Total_CPU_%'] > 30), 'Balance_Load'] = 1
    data.loc[(data['Thread_Count'] > 25) & (data['CPU_Usage_%'] < 120) & (data['Total_CPU_%'] < 75), 'Boost_Parallel'] = 1
    data['CPU_Usage_Diff'] = data.groupby('Process_Name')['CPU_Usage_%'].diff()
    data.loc[(data['CPU_Usage_Diff'] > 80) & (data['Total_CPU_%'] > 60), 'Handle_Burst'] = 1
    data = data.drop(columns=['CPU_Usage_Diff'])

    # Apply labeling conditions for memory conditions
    data.loc[(data['Total_RAM_%'] > 65) & (data['Memory_Usage_%'] > 4), 'Swap_Out'] = 1
    data.loc[(data['VMS_MB'] > 500) & (data['Memory_Usage_%'] < 4), 'Optimize_Memory'] = 1
    data.loc[(data['Memory_Usage_%'] > 2) & (data['VMS_MB'] < 500), 'Increase_VMS'] = 1
    data.loc[(data['VMS_MB'] > 600) & (data['CPU_Usage_%'] < 40), 'Reduce_VMS'] = 1
    data.loc[(data['User_Priority'] > 32) & (data['Free_RAM_%'] > 40), 'Prioritize_RAM'] = 1

    # Save labeled data
    data.to_csv(output_file, index=False)
    print("Label distribution:")
    print(data[['Throttle', 'Prioritize', 'Reduce_Switching', 'Balance_Load', 
                'Boost_Parallel', 'Handle_Burst', 'Swap_Out', 'Optimize_Memory', 
                'Increase_VMS', 'Reduce_VMS', 'Prioritize_RAM']].sum())
    
    return data
