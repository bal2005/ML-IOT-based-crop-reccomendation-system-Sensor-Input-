

# ML-IOT-based-crop-reccomendation-system-Sensor-Input-

This repository contains scripts and files related to sensor data acquisition, processing, and visualization, primarily focusing on NPK sensor readings.

## Project Structure

- **myenv/**: Contains the virtual environment for the project.
- **pyserial/**: Python package for serial communication, primarily used to communicate with the Arduino.
- **static/**: Directory for storing static files, if any.
- **minmaxscaler.pkl**: Pickle file for storing the Min-Max Scaler model used in data normalization.
- **random_forest_model.pkl**: Pickle file for storing the trained Random Forest model.
- **standscaler.pkl**: Pickle file for storing the Standard Scaler model used in data normalization.
- **sensor_data.txt**: Text file containing raw sensor data.
- **sensor_data4.txt**: Another text file with raw sensor data, possibly from a different run or configuration.
- **requirements.txt**: Contains the list of dependencies required to run the project.
- **npk_sensor_file_convert.py**: Script to get readings from the NPK sensor and convert the data into a file.
- **npk_sensor_read.py**: Script to read sensor data from Arduino (used for demonstration purposes).
- **st_ap.py**: Script to display the output, likely involving data visualization or a simple application interface.

## How to Use

1. **Set Up the Environment:**
   - Create and activate a virtual environment (optional but recommended).
   - Install the required dependencies using `pip install -r requirements.txt`.

2. **Getting Sensor Data:**
   - Use `npk_sensor_file_convert.py` to obtain readings from the NPK sensor and save them to a file.
   - For a demonstration of reading sensor data directly from the Arduino, use `npk_sensor_read.py`.

3. **Displaying Output:**
   - Run `st_ap.py` to visualize or display the processed data.

## Dependencies

Make sure all dependencies listed in `requirements.txt` are installed before running any scripts.

