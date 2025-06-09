# for each filename.txt in the directory yolotrain/dataset/labels, copy frames/filename.png to yolotrain/dataset/images
import os

def copy_frames_to_images(labels_dir, source_dir, images_dir):
    # Create the images directory if it doesn't exist
    #os.makedirs(images_dir, exist_ok=True)

    # Iterate through all .txt files in the labels directory
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt") and filename.startswith("dep"):
            # Get the base name without the extension
            base_name = os.path.splitext(filename)[0]
            # Construct the corresponding image file name
            image_file = os.path.join(source_dir, f"{base_name}.png")
            # Check if the image file exists
            if os.path.exists(image_file):
                print(f"Copying {image_file} to {images_dir}")
                # Copy the image file to the images directory
                os.system(f"cp {image_file} {images_dir}")

if __name__ == "__main__":
    labels_dir = "yolotrain/dataset/labels"
    source_dir = "frames"
    images_dir = "yolotrain/dataset/images"
    
    copy_frames_to_images(labels_dir, source_dir, images_dir)                
