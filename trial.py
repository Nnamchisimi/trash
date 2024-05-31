import cv2
import cvzone
import math
import os
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

def detect_objects(image_dir, output_dir, weights_path, classNames):
    results = []

    # Load the YOLOv7 model with the specified weights file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights_path, map_location=device)  # Load the model
    model.eval()

    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            img0 = img.copy()

            # Prepare image for inference
            img = cv2.resize(img, (640, 640))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Perform object detection
            with torch.no_grad():
                pred = model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    # Draw bounding boxes and labels of detections
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        w, h = x2 - x1, y2 - y1

                        # Draw a rectangle with rounded corners around the detected object
                        cvzone.cornerRect(img0, (x1, y1, w, h))

                        # Round the confidence score
                        conf = math.ceil((conf * 100)) / 100

                        # Display class name and confidence score on the image
                        cvzone.putTextRect(img0, f'{classNames[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                        # Print detection results
                        print(f"Image: {filename}, Class: {classNames[int(cls)]}, Confidence: {conf}, Bounding Box: {x1, y1, x2, y2}")

                        # Determine if it's trash or not
                        is_trash = classNames[int(cls)] == "trash"

                        if is_trash:
                            # Calculate the horizontal center of the bounding box
                            center_x = (x1 + x2) / 2
                            image_center_x = img0.shape[1] / 2

                            # Calculate the angle based on the position
                            angle = (center_x / img0.shape[1]) * 180

                            # Determine the direction based on the angle
                            if 0 <= angle <= 45:
                                direction = "turn left"
                            elif 135 <= angle <= 180:
                                direction = "turn right"
                            else:
                                direction = "center"

                            # Append result to the list with direction and angle
                            results.append((filename, classNames[int(cls)], conf, (x1, y1, x2, y2), direction, angle))

                            # Print direction and angle
                            print(f"Direction: {direction}, Angle: {angle}")

            # Save the output image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img0)

            # Display the output image
            cv2.imshow("Image", img0)
            cv2.waitKey(1000)  # Display each image for 1 second

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return results

# List of class names, assuming two classes: "not trash" and "trash"
classNames = ["not trash", "trash"]

# Call the function with your desired parameters
image_dir = r"C:\Users\CIU\Desktop\trash\images"
output_dir = r"C:\Users\CIU\Desktop\trash\output"
weights_path = r"C:\Users\CIU\Desktop\trash\epoch_054.pt"

detection_results = detect_objects(image_dir, output_dir, weights_path, classNames)

# Print detection results
for result in detection_results:
    print(result)
