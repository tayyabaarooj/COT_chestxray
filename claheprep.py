import cv2
import numpy as np
import os

def apply_clahe_to_image(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE contrast normalization to a grayscale image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img)
    return clahe_img

if __name__ == '__main__':

    input_folder = r'/home/intern08/Documents/X_ray_CoT/dataset/images/images_normalized'
    output_folder = r'/home/intern08/Documents/X_ray_CoT/dataset/clahe_images'


    clahe_clip_limit = 2.0
    clahe_tile_grid_size = (8, 8)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            enhanced_image = apply_clahe_to_image(
                input_image_path,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size
            )

            if enhanced_image is not None:
                cv2.imwrite(output_image_path, enhanced_image)
                print(f"Saved CLAHE-enhanced image to: {output_image_path}")
