import cv2
import os
import shutil

# Set the paths for your dataset folder and the folder to store skipped images
dataset_folder = 'D:/Suhrut_Documents/Master_study/2nd_Semester/MSE/Exersice/ImageDataBase/RoadFollowingOldDataset/images'
skipped_folder = 'D:/Suhrut_Documents/Master_study/2nd_Semester/MSE/Exersice/ImageDataBase/skippedimages'
clicked_folder = 'D:/Suhrut_Documents/Master_study/2nd_Semester/MSE/Exersice/ImageDataBase/clickedImages'
delete_folder = 'D:/Suhrut_Documents/Master_study/2nd_Semester/MSE/Exersice/ImageDataBase/deleteImages'
def process_image(image_path):
    image_name = os.path.basename(image_path)
    x, y, uuid = image_name.split("_")

    image = cv2.imread(image_path)
    circle_color = (0, 255, 0)  # Green color
    circle_radius = 5
    circle_thickness = 2
    cv2.circle(image, (int(x), int(y)), circle_radius, circle_color, circle_thickness)

    cv2.imshow("Image", image)
    key = cv2.waitKey(0)

    if key == ord("s"):
        new_image_path = os.path.join(skipped_folder, image_name)
        save_original_image(image_path, new_image_path)
        print(f"Image skipped and saved as {image_name}")

    elif key == ord("e"):
        new_x, new_y = select_coordinates(image)
        new_image_name = f"{new_x}_{new_y}_{uuid}"
        new_image_path = os.path.join(clicked_folder, new_image_name)
        save_modified_image(image_path, new_image_path, new_x, new_y)
        print(f"Image saved as {new_image_name}")
    
    elif key == ord("d"):
        new_image_path = os.path.join(delete_folder, image_name)
        save_original_image(image_path, new_image_path)
        print(f"Image deletion and saved as {image_name}")

    cv2.destroyAllWindows()

def select_coordinates(image):
    clone = image.copy()
    selected_point = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)
            cv2.destroyAllWindows()

    cv2.namedWindow("Select Coordinates")
    cv2.setMouseCallback("Select Coordinates", mouse_callback)

    while selected_point is None:
        cv2.imshow("Select Coordinates", clone)
        cv2.waitKey(1)

    return selected_point

def save_modified_image(original_image_path, new_image_path, new_x, new_y):
    image = cv2.imread(original_image_path)
    image_name = os.path.basename(original_image_path)
    uuid = image_name.split("_")[-1]
    new_image_name = f"{new_x}_{new_y}_{uuid}"
    new_image_path = os.path.join(clicked_folder, new_image_name)
    cv2.imwrite(new_image_path, image)

def save_original_image(original_image_path, new_image_path):
    image = cv2.imread(original_image_path)
    cv2.imwrite(new_image_path, image)

# Get a list of all image files in the dataset folder
image_files = [f for f in os.listdir(dataset_folder) if f.endswith(".jpg") or f.endswith(".png")]
image_path = os.path.join(dataset_folder, "15_83_f0510eea-f4b8-11ed-b7ff-a46bb6069316.jpg")


# Process images one by one
for image_file in image_files:
    image_path = os.path.join(dataset_folder, image_file)
    process_image(image_path)


