import serial
import time

def transmit_uart3(port='/dev/ttyS3', baudrate=9600, transmit_value=25, interval=10, num_repeats=2):
    """
    Transmit the specified value over UART 3 periodically.

    Args:
        port (str): The UART port (default is '/dev/ttyS2' for UART3).
        baudrate (int): The baud rate for UART communication (default is 9600).
        transmit_value (int): The value to transmit (default is 25).
        interval (int): The time in seconds to wait between each set of transmissions (default is 10).
        num_repeats (int): The number of times to transmit the value in each cycle (default is 2).

    Returns:
        None
    """
    try:
        # Initialize the serial connection
        ser = serial.Serial(port, baudrate, timeout=1)

        # Check if the serial connection is open
        if not ser.is_open:
            ser.open()

        print(f"Transmitting {transmit_value} over {port} every {interval} seconds...")

        while True:
            # Transmit the value the specified number of times
            for _ in range(num_repeats):
                # Transmit the value as a byte string (ensure the value is a byte)
                ser.write(bytes([transmit_value]))
                print(f"Transmitted value: {transmit_value}")
                time.sleep(1)  # Wait for 1 second between transmissions

            # Wait for the specified interval before repeating the cycle
            print(f"Waiting for {interval} seconds...\n")
            time.sleep(interval)

    except serial.SerialException as e:
        print(f"Error opening or using the serial port: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("Closed UART connection.")

# Example call to the function (it will start the transmission loop)
transmit_uart3()

