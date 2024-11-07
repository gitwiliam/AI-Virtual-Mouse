#!pip install pyautogui
#!apt-get install -y x11-utils
#!apt-get install -y xvfb
#!pip install mediapipe


import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Define gesture actions
def perform_gesture_action(landmarks):
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index finger tip
    middle_tip = landmarks[12]  # Middle finger tip

    # Calculate distances between thumb and index, thumb and middle
    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
    thumb_middle_distance = calculate_distance(thumb_tip, middle_tip)

    # Set a threshold for detecting gestures
    click_threshold = 20

    # LEFT CLICK: Thumb and index finger close together
    if thumb_index_distance < click_threshold:
        pyautogui.click()
        return "Left Click"

    # RIGHT CLICK: Thumb and middle finger close together
    elif thumb_middle_distance < click_threshold:
        pyautogui.click(button='right')
        return "Right Click"

    # DRAG: Thumb, index, and middle fingers together
    elif thumb_index_distance < click_threshold and thumb_middle_distance < click_threshold:
        pyautogui.mouseDown()
        return "Drag"

    # ZOOM IN: Pinching gesture (distance between index and middle finger decreases)
    elif thumb_index_distance < click_threshold and thumb_middle_distance > 50:
        pyautogui.hotkey('ctrl', '+')  # Simulate zoom-in
        return "Zoom In"

    # ZOOM OUT: Reverse pinching gesture (distance between index and middle finger increases)
    elif thumb_index_distance > 50 and thumb_middle_distance < click_threshold:
        pyautogui.hotkey('ctrl', '-')  # Simulate zoom-out
        return "Zoom Out"

    # VOLUME ADJUSTMENT: Vertical hand motion (use index finger tip y-coordinate)
    elif landmarks[8][1] < screen_height // 3:
        pyautogui.press('volumeup')  # Volume up
        return "Volume Up"
    elif landmarks[8][1] > 2 * screen_height // 3:
        pyautogui.press('volumedown')  # Volume down
        return "Volume Down"

    return None

# Main loop
def main():
    cap = cv2.VideoCapture(0)  # Start video capture
    prev_action = None

    while True:
        success, frame = cap.read()  # Read frame from webcam
        if not success:
            break

        # Flip the frame horizontally for natural hand movement
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (Mediapipe requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmarks
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * screen_width)
                    y = int(lm.y * screen_height)
                    landmarks.append((x, y))

                # Perform action based on detected gesture
                action = perform_gesture_action(landmarks)

                if action and action != prev_action:
                    print(f"Action: {action}")
                    prev_action = action

        # Display the frame
        cv2.imshow("AI Virtual Mouse", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
