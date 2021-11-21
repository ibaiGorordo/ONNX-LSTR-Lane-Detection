import cv2
import pafy
from lstr import LSTR

model_path = "models/model_float32.onnx"

# Initialize video
# cap = cv2.VideoCapture("video.mp4")
videoUrl = "https://youtu.be/2CIxM7x-Clc"
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)

# Initialize lane detection model
lane_detector = LSTR(model_path)

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280,720))

while cap.isOpened():
	try:
		# Read frame from the video
		ret, frame = cap.read()
	except:
		continue

	if ret:	

		# Detect the lanes
		detected_lanes, lane_ids = lane_detector.detect_lanes(frame)
		output_img = lane_detector.draw_lanes(frame)

		cv2.imshow("Detected lanes", output_img)
		out.write(output_img)

	else:
		break

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()