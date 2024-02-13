from PIL import Image
import os


def convert_to_jpg(input_path, output_path):
    try:
        # Open the image file
        with Image.open(input_path) as img:
            # Save the image in JPEG format
            img.convert("RGB").save(output_path, "JPEG")
        print(f"Conversion successful: {output_path}")
        # os.remove(input_path)


    except Exception as e:
        print(f"Error converting image: {str(e)}")