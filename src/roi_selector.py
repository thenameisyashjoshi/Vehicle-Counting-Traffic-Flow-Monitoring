import cv2

IMAGE_PATH = r"D:/CapstoneProject/traffic_frame.jpg"

# Load original image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# Resize to match the final video-processing resolution
image = cv2.resize(image, (1280, 720))

original = image.copy()
points = []

window_name = "ROI Coordinate Selector - 1280x720"


def mouse_callback(event, x, y, flags, param):
    global image, points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        cv2.putText(
            image,
            f"{len(points)}: ({x}, {y})",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

        cv2.imshow(window_name, image)

        print(f"Point {len(points)}: ({x}, {y})")


cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)
cv2.setMouseCallback(window_name, mouse_callback)

print("Image resolution: 1280 x 720")
print("Click the ROI boundary points in order.")
print("Press R to reset all points.")
print("Press C to finish and print coordinates.")
print("Press Q to quit.")

while True:
    cv2.imshow(window_name, image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        points.clear()
        image = original.copy()
        print("Points reset.")

    elif key == ord("c"):
        print("\nROI_POINTS = [")
        for point in points:
            print(f"    {point},")
        print("]")

        break

    elif key == ord("q"):
        break

cv2.destroyAllWindows()

if points:
    print("\nFinal coordinates:")
    print(points)
