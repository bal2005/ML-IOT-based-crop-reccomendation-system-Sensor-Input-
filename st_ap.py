import streamlit as st
import requests
import random
import numpy as np
import pickle

# Weather API Key and Base URL
API_KEY = '091dc1f48fe3c588fd3b05a505f07e71'  # Replace with your actual Weatherstack API key
BASE_URL = "http://api.weatherstack.com/current"

# Load the model and scalers
model = pickle.load(open('random_forest_model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

def get_weather(location):
    """Fetches current weather data for a specified location."""
    params = {
        'access_key': API_KEY,
        'query': location
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if 'current' in data:
            current_weather = data['current']
            temperature = current_weather['temperature']
            humidity = current_weather['humidity']
            return temperature, humidity
        else:
            st.error(f"Error fetching data: {data.get('error', {}).get('info', 'Unknown error')}")
            return None, None

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None

def read_sensor_data_from_file(file_path):
    """Reads sensor data from the text file and returns it as a dictionary."""
    sensor_data = {"SM": 0.0, "N": 0.0, "P": 0.0, "K": 0.0}
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key in sensor_data:
                        try:
                            sensor_data[key] = float(value)
                        except ValueError as ve:
                            st.error(f"ValueError: {ve} while processing {key} data")
    except FileNotFoundError:
        st.error("Sensor data file not found.")
    except Exception as e:
        st.error(f"Error reading sensor data: {e}")
    
    return sensor_data

def main():
    st.title('Sensor Data and Weather Information')

    # Path to the sensor data file
    file_path = 'D:/Inventify/sensor_data4.txt'  # Adjust the path if necessary

    # Read sensor data from the file
    sensor_data = read_sensor_data_from_file(file_path)
    sm = sensor_data.get('SM', 'Not Available')
    n = sensor_data.get('N', 'Not Available')
    p = sensor_data.get('P', 'Not Available')
    k = sensor_data.get('K', 'Not Available')

    # Display sensor data
    st.write("Soil Moisture Level: ", sm)
    st.write("Nitrogen: ", n)
    st.write("Phosphorus: ", p)
    st.write("Potassium: ", k)

    # Generate random values for rainfall and pH
    
    rainfall = random.uniform(0, 300)  # Rainfall in cm
    ph = random.uniform(5, 7)  # pH value

    st.write(f"Rainfall: {rainfall:.2f} cm")
    st.write(f"pH: {ph:.2f}")

    # Initialize weather variables
    temperature = 0.0
    humidity = 0

    # Input field for location
    location = st.text_input('Enter location:')

    if st.button('Get Weather'):
        if location:
            temp, hum = get_weather(location)
            if temp is not None and hum is not None:
                temperature = temp
                humidity = hum
                st.write(f"Temperature: {temperature}°C")
                st.write(f"Humidity: {humidity}%")
            else:
                st.error("Failed to fetch weather data.")
        else:
            st.error("Please enter a location.")
    
    if st.button('Predict Crop'):
        # Use sensor data for N, P, K, and PH values
        N = sensor_data["N"]
        P = sensor_data["P"]
        K = sensor_data["K"]

        # Use rainfall and pH from the random values generated
        feature_list = [N, P, K, temperature, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scaling the input features
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Predicting probabilities
        probabilities = model.predict_proba(final_features)[0]

        # Sorting and selecting top 3 indices
        top_3_indices = np.argsort(probabilities)[-3:][::-1]

        # Mapping indices to crop names
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        result_image = {
            1: "static/crops/Rice.jpeg", 2: "static/crops/Maize.jpeg", 3: "static/crops/Jute.jpeg",
            4: "static/crops/Cotton.jpeg", 5: "static/crops/Coconut.jpeg", 6: "static/crops/Papaya.jpeg",
            7: "static/crops/Orange.jpeg", 8: "static/crops/Apple.jpeg", 9: "static/crops/Muskmelon.jpeg",
            10: "static/crops/Watermelon.jpeg", 11: "static/crops/Grapes.jpeg", 12: "static/crops/Mango.jpeg",
            13: "static/crops/Banana.jpeg", 14: "static/crops/Pomegranate.jpeg", 15: "static/crops/Lentil.jpeg",
            16: "static/crops/Blackgram.jpeg", 17: "static/crops/Mungbean.jpeg", 18: "static/crops/Mothbeans.jpeg",
            19: "static/crops/Pigeonpeas.jpeg", 20: "static/crops/Kidneybeans.jpeg", 21: "static/crops/Chickpea.jpeg",
            22: "static/crops/Coffee.jpeg"
        }

        crop_description = {
            "Rice": "Rice is predominantly cultivated in Asia, especially in regions like China, India, and Southeast Asia. It thrives in clayey, loamy soils that retain water well.",
            "Maize": "Maize, or corn, is widely grown in the United States, Brazil, and China. It prefers well-drained, fertile loamy soils rich in organic matter.",
            "Jute": "Jute is mainly cultivated in Bangladesh and India, particularly in the Ganges Delta. It grows best in alluvial soil with a high clay content.",
            "Cotton": "Cotton is densely grown in India, the United States, and China. It favors deep, well-drained sandy loam soils with a slightly acidic to neutral pH.",
            "Coconut": "Coconut palms are most commonly found in tropical coastal regions like Indonesia, the Philippines, and India. They grow well in sandy, loamy, or alluvial soils that are well-drained.",
            "Papaya": "Papaya is extensively grown in India, Brazil, and Mexico. It thrives in well-drained sandy loam or alluvial soils rich in organic matter.",
            "Orange": "Oranges are primarily cultivated in Brazil, the United States (especially Florida), and China. They prefer well-drained sandy loam soils rich in organic content.",
            "Apple": "Apples are widely grown in temperate regions like the United States, China, and Europe. They thrive in well-drained loamy soils with a slightly acidic pH.",
            "Muskmelon": "Muskmelon is grown in warmer climates such as in India and the United States. It prefers sandy loam soils that are well-drained and rich in organic matter.",
            "Watermelon": "Watermelon is cultivated in warm regions like China, Turkey, and the United States. It grows best in sandy loam soils with good drainage and moderate organic content.",
            "Grapes": "Grapes are primarily cultivated in Mediterranean climates like those found in Italy, Spain, and France. They thrive in well-drained loamy or sandy loam soils with good fertility.",
            "Mango": "Mangoes are extensively grown in India, Thailand, and Mexico. They favor well-drained alluvial or loamy soils rich in organic matter.",
            "Banana": "Bananas are widely cultivated in tropical regions such as India, Brazil, and Ecuador. They thrive in well-drained loamy soils with high organic content.",
            "Pomegranate": "Pomegranates are primarily grown in India, Iran, and the Mediterranean. They prefer well-drained sandy or loamy soils with a neutral to slightly alkaline pH.",
            "Lentil": "Lentils are commonly grown in India, Turkey, and Canada. They prefer well-drained loamy soils with a neutral to slightly acidic pH.",
            "Blackgram": "Blackgram is mainly cultivated in India and Myanmar. It thrives in well-drained loamy soils with moderate fertility.",
            "Mungbean": "Mungbean is widely grown in India, China, and Southeast Asia. It prefers well-drained loamy soils with good fertility.",
            "Mothbeans": "Mothbeans are grown in India and parts of Africa. They thrive in well-drained sandy loam soils with a moderate organic content.",
            "Pigeonpeas": "Pigeonpeas are primarily cultivated in India, Africa, and the Caribbean. They grow well in well-drained loamy soils with good fertility.",
            "Kidneybeans": "Kidneybeans are commonly grown in India, the United States, and Brazil. They prefer well-drained loamy soils with a slightly acidic to neutral pH.",
            "Chickpea": "Chickpeas are widely cultivated in India, Turkey, and Australia. They thrive in well-drained loamy soils with a neutral to slightly acidic pH.",
            "Coffee": "Coffee is grown in tropical regions like Brazil, Colombia, and Ethiopia. It prefers well-drained, slightly acidic loamy soils rich in organic matter."
        }

        st.write("Top 3 Crop Recommendations:")
        for idx in top_3_indices:
            crop_name = crop_dict.get(idx , "Unknown Crop")
            st.write(f"Crop: {crop_name}")
            st.write(crop_description.get(crop_name, "Description not available."))
            image_path = result_image[idx]
            st.image(image_path, caption=crop_name, use_column_width=True)

if __name__ == "__main__":
    main()
