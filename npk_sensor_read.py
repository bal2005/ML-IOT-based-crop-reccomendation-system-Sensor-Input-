import serial
import asyncio

# Configure serial connection
ser = serial.Serial('COM6', 9600)  # Replace 'COM3' with your port and 9600 with your baud rate

async def read_serial():
    while True:
        try:
            data = ser.readline().decode('utf-8').strip()  # Read and decode the serial data
            print(data)  # Print the received data to the console
        except Exception as e:
            print(f"Error: {e}")

# Run the async function
asyncio.run(read_serial())