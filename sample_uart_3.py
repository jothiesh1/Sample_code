import requests
import time
import serial
import serial.tools.list_ports
import threading

def find_serial_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'USB' in port.description:
            return port.device
    return None

def send_serial_data(value, times=5, interval=10):
    port = find_serial_port()
    if port is None:
        print("No suitable serial port found.")
        return
    
    baud_rate = 9600
    try:
        with serial.Serial(port, baud_rate, timeout=1) as ser:
            print(f"Connected to {port} at {baud_rate} baud.")
            time.sleep(1)  # Wait for the connection to stabilize
            
            for i in range(times):
                ser.write(bytes([value]))  # Send as a bytes object
                print(f"Sent: {value} ({i + 1}/{times})")
                time.sleep(interval)  # Wait for the specified interval
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except Exception as e:
        print(f"An error occurred while sending data: {e}")
        
        

def main():
	threading.Thread(target=send_serial_data, args=(100, 5, 10)).start()
