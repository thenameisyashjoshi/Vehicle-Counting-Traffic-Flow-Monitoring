import cv2

video_path = r"D:/CapstoneProject/Videos/vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

if ret:
    cv2.imwrite("traffic_frame.jpg", frame)
    print("Frame saved successfully!")
    print("Frame dimensions:", frame.shape[1], "x", frame.shape[0])
else:
    print("Could not read video.")

cap.release()
