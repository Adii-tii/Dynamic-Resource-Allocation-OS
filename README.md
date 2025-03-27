# Adaptive Resource Allocation in Multi-Programming Systems

## Brief

This project implements a machine learning-based system for dynamic resource allocation in multi-programming environments. It monitors system resources (CPU, memory, etc.), predicts optimal resource allocation strategies, and provides actionable insights to improve system performance. The system uses a pre-trained Random Forest model to predict strategies such as throttling, prioritizing, or swapping out processes based on real-time system metrics.

The project includes a web-based user interface built with Gradio, featuring multiple tabs for monitoring, prediction, dataset analysis, model tuning, and system insights.

## Features

- Real-time system monitoring with dynamic graphs for CPU and memory usage.
- Prediction of resource allocation strategies based on user-defined inputs.
- Analysis of uploaded datasets to predict strategies for multiple processes.
- Model tuning with adjustable hyperparameters (n_estimators, max_depth, min_samples_split).
- System insights with recommendations based on current resource usage.

## Project Structure

- `main.py`: The main script containing the Gradio UI and all functionality.
- `model.pkl`: Pre-trained Random Forest model for predictions.
- `model_assets.pkl`: Contains the model, scaler, and label encoder used for preprocessing.
- `requirements.txt`: List of Python dependencies required to run the project.
- `resource_data_augmented_new3.csv`: The dataset used for training the model (not included in the repository due to size).

## Prerequisites

- Python 3.8 or higher
- Git
- A Kaggle account (if running on Kaggle) or a local environment with sufficient resources

## Setup Instructions

1. **Clone the Repository**
   ```
   git clone https://github.com/your-username/dynamic-resource-allocation.git
   cd dynamic-resource-allocation
   ```

2. **Install Dependencies**
   Create a virtual environment and install the required packages:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare the Model and Assets**
   Ensure the following files are in the `/kaggle/working/` directory (or adjust paths in the code):
   - `model.pkl`: The pre-trained model.
   - `model_assets.pkl`: Contains the model, scaler, and label encoder.
   If running on Kaggle, upload these files to your Kaggle working directory.

4. **Prepare the Dataset**
   The project uses `resource_data_augmented_new3.csv` for retraining the model in the Model Settings tab. Place this file in `/kaggle/input/resource-data-updated/` or update the path in the code.

5. **Run the Application**
   Execute the main script to launch the Gradio UI:
   ```
   python main.py
   ```
   Follow the link provided by Gradio to access the interface in your browser.

## Usage

The Gradio UI consists of five tabs:

1. **Dashboard**
   - Displays a real-time graph of CPU and memory usage on the left.
   - Shows a table of active processes (top 20 by CPU usage) on the right.
   - Updates automatically every second.

2. **Action Prediction**
   - Input system metrics (e.g., CPU usage, RAM usage) using sliders.
   - Select a process name from the dropdown.
   - Click "Submit" to predict resource allocation strategies (e.g., "Throttle", "Swap_Out").

3. **Dataset Analysis**
   - Upload a CSV file with system metrics for multiple processes.
   - The system predicts strategies for each row and adds them as new columns.
   - View the processed dataset and download it as a CSV file.

4. **Model Settings**
   - Adjust hyperparameters (n_estimators, max_depth, min_samples_split) of the Random Forest model.
   - Click "Submit" to retrain the model with the new settings.
   - The updated model is saved to `model_assets.pkl`.

5. **System Insights**
   - Provides insights based on current system metrics (e.g., high CPU usage, memory usage).
   - Suggests strategies to improve performance.

## Dataset Format

The dataset (`resource_data_augmented_new3.csv`) should contain the following columns:
- **Input Features**: `Total_CPU_%`, `Total_RAM_%`, `Free_RAM_%`, `CPU_Usage_%`, `CPU_Time_s`, `Context_Switches`, `Memory_Usage_%`, `VMS_MB`, `Thread_Count`, `User_Priority`, `Process_Name`
- **Target Labels**: `Throttle`, `Prioritize`, `Reduce_Switching`, `Balance_Load`, `Boost_Parallel`, `Handle_Burst`, `Swap_Out`, `Optimize_Memory`, `Increase_VMS`, `Reduce_VMS`, `Prioritize_RAM`

## Dependencies

See `requirements.txt` for a full list of dependencies. Key libraries include:
- pandas
- numpy
- scikit-learn
- matplotlib
- psutil
- gradio
- pickle

## Limitations

- The project assumes the pre-trained model and assets are available in the specified paths.
- Real-time monitoring may be resource-intensive on low-spec systems.
- The Gradio UI may have limited styling options without additional customization.

## Contributing

Contributions are welcome. To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request.


## Contact

For questions or issues, please open an issue on GitHub or contact sableaditi8@gmail.com

---
