from ultralytics import YOLO
from IPython.display import display
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

'''
# 1. Load the model
model = YOLO("yolov8s.pt")  # Or your custom model

# 2. Prepare the image
image_path = "frames/frame_0001.png"
# 3. Run inference
img = cv2.imread(image_path)  # Load the image
img = cv2.resize(img, (640, 640))  # Resize to 640x640
results = model.predict(img)


# 4. Post-process and visualize
for result in results:
    for box in result.boxes:
        # Access bounding box coordinates, confidence score, and class label
        bbox = box.xyxy[0].tolist()  # xmin, ymin, xmax, ymax
        conf = box.conf[0].item()
        class_id = box.cls[0].item()
        # Draw bounding box and label on the image
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0), thickness=2)
        cv2.putText(img, f"{model.names[class_id]} {conf:.2f}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("YOLO Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# for each filename.txt in the directory yolotrain/dataset/labels, copy frames/filename.png to yolotrain/dataset/images
import os

def copy_frames_to_images(labels_dir, source_dir, images_dir):
    # Create the images directory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)

    # Iterate through all .txt files in the labels directory
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            # Get the base name without the extension
            base_name = os.path.splitext(filename)[0]
            # Construct the corresponding image file name
            image_file = os.path.join(source_dir, f"{base_name}.png")
            # Check if the image file exists
            if os.path.exists(image_file):
                print(f"Copying {image_file} to {images_dir}")
                # Copy the image file to the images directory
                os.system(f"cp {image_file} {images_dir}")

from ultralytics import YOLO
import cv2

def is_bbox_inside(inner, outer, threshold=0.8):
    # Compute intersection
    xA = max(inner[0], outer[0])
    yA = max(inner[1], outer[1])
    xB = min(inner[2], outer[2])
    yB = min(inner[3], outer[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])

    return (inter_area / inner_area) >= threshold

def check_place_success(image_path, pick_class, place_class, model_path='yolotrain/runs/detect/train/weights/best.pt'):
    model = YOLO(model_path)
    results = model.predict(image_path, imgsz=640, verbose=True)[0]

    annotated_img = results.plot()  # Visualize YOLO detections
    plt.imshow(annotated_img)
    plt.axis('off')
    plt.savefig("yolo_output2.png")

    # Parse bounding boxes
    pick_boxes = []
    place_boxes = []
    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if cls_name == pick_class:
            print(f"Pick box: {x1}, {y1}, {x2}, {y2}")
            pick_boxes.append([x1, y1, x2, y2])
        elif cls_name == place_class:
            print(f"Place box: {x1}, {y1}, {x2}, {y2}")
            place_boxes.append([x1, y1, x2, y2])

    # Evaluate condition: pick is 80% inside any place box
    for pick_box in pick_boxes:
        for place_box in place_boxes:
            if is_bbox_inside(pick_box, place_box):
                return True
    return False

if __name__ == "__main__":
    #labels_dir = "yolotrain/dataset/labels"
    #source_dir = "frames"
    #images_dir = "yolotrain/dataset/images"
    
    #copy_frames_to_images(labels_dir, source_dir, images_dir)                
    # for each filename in images_dir, 

    image_path = "./img.png"

    #img = Image.open(image_path).convert('RGB')
    img = Image.open(image_path)
    image = img.resize((640, 480))  # Resize to 640x640
    image = image.convert('RGB')  # Convert to RGB if needed
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

    success = check_place_success(image, pick_class="scrub", place_class="bowl")
    #print(f"Place Success: {success}")
    #wait for user keypress
    # display using matplotlib
    
    #plt.figure(figsize=(10, 10))


    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.title(f"Place Success: {success}")
    #plt.axis('off')
    #plt.show(block=False)
    #input("Press Enter to close the image...")
    #plt.close()    

    '''
    for filename in os.listdir("yolotrain/dataset/images"):
        if filename.endswith(".png"):
            image_path = os.path.join("yolotrain/dataset/images", filename)
            print(f"Processing {image_path}")
            # Check if the pick is inside the place
            success = check_place_success(image_path, pick_class="scrub", place_class="bowl")
            print(f"Place Success: {success}")
            #wait for user keypress
            # display using matplotlib
            plt.figure(figsize=(10, 10))
            image = cv2.imread(image_path)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Place Success: {success}")
            plt.axis('off')
            plt.show(block=False)
            plt.pause(5)  # Pause for 2 seconds before closing
            plt.close()
    '''