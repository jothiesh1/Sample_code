import serial
import threading
import time  # Add this import for time.sleep
UART_PORT = '/dev/ttyS3'  # Update based on your platform
BAUD_RATE = 9600

# Initialize serial connection
def init_serial_connection():
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Connected to {UART_PORT} at {BAUD_RATE} baud.")
        return ser
    except serial.SerialException as e:
        print(f"[ERROR] Serial connection failed: {e}")
        return None

def send_serial_data(value, num_retries=2, retry_delay=10):
    """ Send data over UART in a separate thread """
    try:
        ser = init_serial_connection()  # Open serial connection inside the function
        if ser is not None:
            byte_data = value.to_bytes(1, byteorder='little')  # Convert the value to a byte
            ser.write(byte_data)
            print(f"[INFO] Sent: {value}")
            ser.close()  # Close the connection after sending
        else:
            print("[ERROR] Serial connection is not available.")
    except Exception as e:
        print(f"[ERROR] Failed to send data: {e}")
        # Optionally handle retries
        if num_retries > 0:
            print(f"[INFO] Retrying... {num_retries} attempts left.")
            time.sleep(retry_delay)
            send_serial_data(value, num_retries - 1, retry_delay)

# Start the sending process in a new thread
threading.Thread(target=send_serial_data, args=(25,)).start()  # Send 25 when prolonged eye closure is detected

