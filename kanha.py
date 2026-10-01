import cv2
import turtle
import numpy as np

# ----------------------------------------------------
# 1. Image Loading and Centerline Extraction
# ----------------------------------------------------
IMAGE_NAME = "KANHA.JPG"

img_color = cv2.imread(IMAGE_NAME)
if img_color is None:
    print(f"Error: '{IMAGE_NAME}' not found! Place the image in the same directory.")
    exit()

target_height = 650
aspect_ratio = img_color.shape[1] / img_color.shape[0]
target_width = int(target_height * aspect_ratio)
img_resized = cv2.resize(img_color, (target_width, target_height))

# Convert to grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Threshold to separate the neon lines from dark background
_, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

# Morphological skeletonization to get clean single-pixel centerline curves
skeleton = np.zeros(binary.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
temp_bin = binary.copy()

while True:
    eroded = cv2.erode(temp_bin, element)
    opened = cv2.dilate(eroded, element)
    subset = cv2.subtract(temp_bin, opened)
    skeleton = cv2.bitwise_or(skeleton, subset)
    temp_bin = eroded.copy()
    if cv2.countNonZero(temp_bin) == 0:
        break

# Extract natural curved contours from skeleton (NO polygon approximation)
contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Draw long lines first for clean visual flow
contours = sorted(contours, key=lambda c: len(c), reverse=True)

# ----------------------------------------------------
# 2. Turtle Screen and Cursor Configuration
# ----------------------------------------------------
screen = turtle.Screen()
screen.setup(width=target_width + 80, height=target_height + 80)
screen.bgcolor("#000000")
screen.title("Shree Krishna Neon Outline Sketch")
screen.colormode(255)

# Fast and smooth animation
screen.tracer(10, 0)

t = turtle.Turtle()
t.shape("classic")
t.pensize(2)
t.speed(0)

offset_x = -target_width // 2
offset_y = target_height // 2

# ----------------------------------------------------
# 3. Live Cursor Drawing
# ----------------------------------------------------
for contour in contours:
    # Skip negligible noise
    if len(contour) < 6:
        continue

    # Move cursor to start of curve
    start_x, start_y = contour[0][0]
    t.penup()
    t.goto(start_x + offset_x, offset_y - start_y)
    t.pendown()

    # Step by 2 pixels to keep curves smooth and speed fast
    for i in range(0, len(contour), 2):
        px, py = contour[i][0]
        turtle_x = px + offset_x
        turtle_y = offset_y - py

        # Extract true pixel color from original neon image
        b, g, r = img_resized[py, px]
        if r < 35 and g < 35 and b < 35:
            r, g, b = 0, 190, 255
            
        t.pencolor(int(r), int(g), int(b))
        t.goto(turtle_x, turtle_y)

screen.update()
t.hideturtle()
print("Sketch completed successfully!")

screen.mainloop()