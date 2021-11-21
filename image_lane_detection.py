import cv2
from imread_from_url import imread_from_url
from lstr import LSTR

model_path = "models/model_float32.onnx"

# Initialize lane detection model
lane_detector = LSTR(model_path)

# Read RGB image
image = imread_from_url("https://live.staticflickr.com/1067/1475776461_f9adc2fee9_o_d.jpg")

# Detect the lanes
detected_lanes, lane_ids = lane_detector.detect_lanes(image)
output_img = lane_detector.draw_lanes(image)

# Draw estimated depth
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL) 
cv2.imshow("Detected lanes", output_img)
cv2.waitKey(0)

cv2.imwrite("output.jpg",output_img)