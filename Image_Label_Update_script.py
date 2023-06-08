import os
import pandas as pd
import cv2

def draw_circle(image, x, y):
    # Draw a green circle on the image at the specified coordinates
    circle_color = (0, 255, 0)  # Green color (BGR format)
    circle_radius = 5
    circle_thickness = 2
    cv2.circle(image, (x, y), circle_radius, circle_color, circle_thickness)

def update_image_name(image_name, x, y):
    # Update the image name with the new x and y coordinates
    updated_name = f"{x}_{y}_{image_name.split('_')[-1]}"
    return updated_name

def process_image(image_path):
    # Extract the x and y coordinates from the image name
    image_name = os.path.basename(image_path)
    x_coordinate, y_coordinate = image_name.split("_")[-3:-1]
    x_coordinate = int(x_coordinate)
    y_coordinate = int(y_coordinate)

    # Load the image
    image = cv2.imread(image_path)

    # Create a window to display the image
    cv2.namedWindow("Image with Circle")
    draw_circle(image, x_coordinate, y_coordinate)
    def mouse_callback(event, x, y, flags, param):
        nonlocal x_coordinate, y_coordinate

        if event == cv2.EVENT_LBUTTONDOWN:
            # Update the coordinates when the left mouse button is clicked
            x_coordinate = x
            y_coordinate = y

            origImage = image
            # Draw a circle at the updated coordinates
            draw_circle(image, x_coordinate, y_coordinate)

            # Display the updated image
            cv2.imshow("Image with Circle", image)

            # Update the image name with the new coordinates
            updated_image_name = update_image_name(image_name, x_coordinate, y_coordinate)
            print("Updated image name:", updated_image_name)

            # Save the updated image
            output_folder = "D:/Suhrut_Documents/Master_study/2nd_Semester/MSE/ProjectDocuments/ImageDB/UpdatedFolder"
            os.makedirs(output_folder, exist_ok=True)
            output_image_path = os.path.join(output_folder, updated_image_name)
            cv2.imwrite(output_image_path, origImage)
            print("Updated image saved successfully.")

            # Close the current image window
            cv2.destroyWindow("Image with Circle")

    # Set the mouse callback function
    cv2.setMouseCallback("Image with Circle", mouse_callback)

    # Display the image
    cv2.imshow("Image with Circle", image)
    cv2.waitKey(0)

# Specify the folder path containing the images
folder_path = "D:/Suhrut_Documents/Master_study/2nd_Semester/MSE/ProjectDocuments/ImageDB/images/apex"

# Get a list of image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Process each image one by one
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    print("Processing image:", image_path)
    process_image(image_path)
    print()

print("All images processed successfully.")

"""

def extract_values_from_filename(filename):
    # Extract x, y, and UUID from the filename
    x, y, uuid = filename.split("_", 2)
    return x, y, uuid

def process_folder(folder_path, output_file):
    # Create an empty DataFrame to store the values
    df = pd.DataFrame(columns=["x", "y", "uuid"])

    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Process each image one by one
    for image_file in image_files:
        x, y, uuid = extract_values_from_filename(image_file)

        # Add the values to the DataFrame
        df = df.append({"x": x, "y": y, "uuid": uuid}, ignore_index=True)

    # Save the DataFrame to an Excel sheet
    df.to_excel(output_file, index=False)
    print("Values saved to", output_file)

# Specify the input folder path
folder_path = "D:/Suhrut_Documents/Master_study/2nd_Semester/MSE/ProjectDocuments/ImageDB/UpdatedFolder"

# Specify the output Excel file path
output_file = "D:/Suhrut_Documents/Master_study/2nd_Semester/MSE/ProjectDocuments/ImageDB/Update.xlsx"

# Process the folder and save the values to the Excel sheet
process_folder(folder_path, output_file)
"""