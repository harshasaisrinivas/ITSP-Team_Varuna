from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import RPi.GPIO as GPIO
import smbus2
from time import sleep
from RPLCD.i2c import CharLCD
import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time

# Use the pigpio pin factory for servos
factory = PiGPIOFactory()

# Initialize servos
servo1 = Servo(16, pin_factory=factory)
servo2 = Servo(26, pin_factory=factory)

# Motor Driver Pins
IN1 = 17
IN2 = 27
IN3 = 22
IN4 = 5
ENA = 13
ENB = 6

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup Motor Driver Pins
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# Initialize PWM for Motor Driver
pwm_A = GPIO.PWM(ENA, 1000)
pwm_B = GPIO.PWM(ENB, 1000)
pwm_A.start(0)
pwm_B.start(0)

# Define the I2C bus and LCD address
i2c_port = 1
lcd_address = 0x27

# Initialize the LCD
lcd = CharLCD('PCF8574', lcd_address, port=i2c_port,
              cols=16, rows=2, dotsize=8,
              charmap='A00', auto_linebreaks=True,
              backlight_enabled=True)

# Initialize Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLO model
model = YOLO('best3.pt')

# Load class names
with open("best3label.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

# Ensure 'bottle' is in the class list
assert 'bottle' in class_list, "Bottle class not found in class list"

# Camera specifications
frame_width = 640
frame_height = 480
vertical_fov = 48.8  # Vertical field of view in degrees (depends on your camera)

# Ultrasonic Sensor Pins
TRIG = 23
ECHO = 24

# Setup Ultrasonic Sensor Pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

rotation_value1 = 0.24  # Value for 60 degrees clockwise
rotation_value2 = 0.16  # Value for 60 degrees counterclockwise

def stop():
    GPIO.output(IN1, False)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, False)
    pwm_A.ChangeDutyCycle(0)
    pwm_B.ChangeDutyCycle(0)
    print("Motors stopped")

def rotate_by_angle(angle):
    # Calculate sleep duration based on angle
    sleep_time = abs(angle / 30) * 0.75
    
    if angle > 0:
        # Rotate clockwise
        GPIO.output(IN1, True)
        GPIO.output(IN2, False)
        GPIO.output(IN3, True)
        GPIO.output(IN4, False)
    else:
        # Rotate counterclockwise
        GPIO.output(IN1, False)
        GPIO.output(IN2, True)
        GPIO.output(IN3, False)
        GPIO.output(IN4, True)
    
    pwm_A.ChangeDutyCycle(95)
    pwm_B.ChangeDutyCycle(95)
    time.sleep(sleep_time)
    stop()

def distance():
    # Set Trigger to HIGH
    GPIO.output(TRIG, True)

    # Set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    StartTime = time.time()
    StopTime = time.time()

    # Save StartTime
    while GPIO.input(ECHO) == 0:
        StartTime = time.time()

    # Save time of arrival
    while GPIO.input(ECHO) == 1:
        StopTime = time.time()

    # Time difference between start and arrival
    TimeElapsed = StopTime - StartTime

    # Multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2

    return distance

def forward(duration=None):
    GPIO.output(IN1, True)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, True)
    pwm_A.ChangeDutyCycle(80)
    pwm_B.ChangeDutyCycle(80)
    print("Motors running forward")
    if duration:
        time.sleep(duration)
        stop()

def main():
    count = 0
    angle = 0.0
    previous_length = 0.0

    try:
        while True:
            im = picam2.capture_array()
            
            count += 1
            if count % 3 != 0:
                continue
            
            im = cv2.flip(im, -1)
            results = model.predict(im)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            
            bottle_found = False
            current_length = 0.0
            
            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = class_list[d]
                
                if c == 'bottle':
                    bottle_found = True
                    # Calculate center of the bottle
                    bottle_center_x = (x1 + x2) / 2
                    bottle_center_y = (y1 + y2) / 2
                    
                    # Calculate the vertical angle
                    delta_y = bottle_center_y - (frame_height / 2)
                    angle = (delta_y / frame_height) * vertical_fov
                    
                    # Calculate the length of the bottle
                    current_length = y2 - y1
                    print(f'Current bottle length: {current_length}')
                    
                    # Draw rectangle and text
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(im, f'{c} {angle:.2f}°', (x1, y1), 1, 1)
            
            if not bottle_found:
                angle = 0.0
                lcd.clear()
                lcd.write_string('BottleFound: False')
                print('Bottle not found, rotating...')
                rotate_by_angle(2)  # Rotate by 2 degrees if no bottle is detected
                continue
            
            # Display status on the LCD
            lcd.clear()
            lcd.write_string(f'BottleFound: True')
            lcd.cursor_pos = (1, 0)
            lcd.write_string(f'Angle: {angle:.2f}°')
            print(f'Bottle found, angle: {angle:.2f}°')

            # Rotate to align with the bottle and continue until the length decreases
            while current_length > previous_length:
                rotate_by_angle(angle)
                previous_length = current_length
                print(f'Rotating by angle: {angle:.2f}°')
                
                # Re-capture the frame and update the length
                im = picam2.capture_array()
                im = cv2.flip(im, -1)
                results = model.predict(im)
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")
                
                for index, row in px.iterrows():
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    d = int(row[5])
                    c = class_list[d]
                    
                    if c == 'bottle':
                        current_length = y2 - y1
                        print(f'Updated bottle length: {current_length}')
                        break
            
            # Move forward towards the bottle
            stop()
            forward(1.8)  # Adjust duration based on your bot's speed
            lcd.clear()
            lcd.write_string('Moving forward')
            print('Moving forward towards the bottle')
            
            # Stop when the bot is between 15 cm and 35 cm from the object
            if 15 < distance() < 35:
                stop()
                lcd.clear()
                lcd.write_string('Within 15-35cm')
                print("Within 15-35 cm. Stopping.")
                break

        # Rotate both servos to 60 degrees clockwise
        servo1.value = rotation_value1
        servo2.value = -rotation_value2
        time.sleep(1.7)

        # Rotate both servos to 60 degrees counterclockwise (return to original position)
        servo1.value = -rotation_value1
        servo2.value = rotation_value2
        time.sleep(1.7)

        # Return both servos to the middle position (0 degrees)
        servo1.mid()
        servo2.mid()
        
    except KeyboardInterrupt:
        print("Program terminated")
    
    finally:
        # Cleanup
        GPIO.cleanup()
        stop()
        lcd.clear()
        lcd.write_string('Cleanup done')
        print("Cleanup completed")

if _name_ == "_main_":
    main()
