import cv2
import os
import random
import numpy as np


def get_image_size():
    # Return standard gesture dimensions directly to prevent reliance on hardcoded images
    return (50, 50)


# Determine absolute path to the 'gestures' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
gestures_dir = os.path.join(script_dir, "gestures")

# Fallback if gestures folder is located in parent directory
if not os.path.exists(gestures_dir):
    gestures_dir = os.path.join(script_dir, "..", "gestures")

if not os.path.exists(gestures_dir):
    raise FileNotFoundError(f"Could not locate 'gestures' folder near: {script_dir}")

# Get and sort numerical gesture subfolders (e.g., '0', '1', '2', ...)
gestures = [f for f in os.listdir(gestures_dir) if f.isdigit()]
gestures.sort(key=int)

if not gestures:
    raise FileNotFoundError(f"No gesture folders found in {gestures_dir}")

image_y, image_x = get_image_size()

begin_index = 0
end_index = 5

if len(gestures) % 5 != 0:
    rows = int(len(gestures) / 5) + 1
else:
    rows = int(len(gestures) / 5)

full_img = None

for i in range(rows):
    col_img = None

    # Determine slice range for current row safely
    current_batch = gestures[begin_index : min(end_index, len(gestures))]

    for folder_num in current_batch:
        folder_path = os.path.join(gestures_dir, folder_num)
        available_images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

        # Pick a random existing image, or construct a blank image if empty
        if available_images:
            random_img_name = random.choice(available_images)
            img_path = os.path.join(folder_path, random_img_name)
            img = cv2.imread(img_path, 0)
        else:
            img = None

        if img is None:
            img = np.zeros((image_y, image_x), dtype=np.uint8)

        # Horizontal stack for row
        if col_img is None:
            col_img = img
        else:
            col_img = np.hstack((col_img, img))

    # Pad remaining columns with blank space if last row has fewer than 5 items
    remaining_cols = 5 - len(current_batch)
    for _ in range(remaining_cols):
        blank = np.zeros((image_y, image_x), dtype=np.uint8)
        col_img = np.hstack((col_img, blank))

    begin_index += 5
    end_index += 5

    # Vertical stack across rows
    if full_img is None:
        full_img = col_img
    else:
        full_img = np.vstack((full_img, col_img))

cv2.imshow("gestures", full_img)
cv2.imwrite("full_img.jpg", full_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
