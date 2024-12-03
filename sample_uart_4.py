import requests
import time
import serial
import serial.tools.list_ports
import threading

def send_serial_data(value, times=2, interval=10):
    try:
	UART_PORT ='/dev/ttyS3'
        with serial.Serial(UART_PORT, BAUD_RATE=9600, timeout=1) as ser:
            print(f"Connected to {UART_PORT} at {BAUD_RATE} baud.")
            ser.flushInput()
            ser.flushOutput()
            #time.sleep(1)  # Wait for the connection to stabilize

            for i in range(times):
                # Ensure we are sending the integer 10 as 8-byte (64-bit)
                # Converting integer value 10 to an 8-byte representation
                byte_data = value.to_bytes(1, byteorder='little')  # 8 bytes for 64-bit integer
                
                ser.write(byte_data)
                print(f"Sent: {value} ({i + 1}/{times})")
                time.sleep(interval)  # Wait for the specified interval
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except Exception as e:
        print(f"An error occurred while sending data: {e}")
        
        

def main():
	threading.Thread(target=send_serial_data, args=(15, 2, 10)).start()
