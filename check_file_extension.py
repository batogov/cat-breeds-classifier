import imghdr
import os
from PIL import Image

def get_list_dir():
    list_dir = os.listdir()
    
    if '.DS_STORE' in list_dir:
        list_dir.remove('.DS_Store')
    return list_dir

def convert_to_jpeg(image_path):
    """ Converts the image file to jpeg and removes the original"""
    assert os.path.isfile(image_path)

    try:
        with Image.open(image_path) as img:
            output_path = os.path.splitext(image_path)[0] + '.jpeg'
            unaccepted_color_modes = Image.MODES.copy()
            unaccepted_color_modes.remove("RGB") 
            if img.mode in unaccepted_color_modes:
                img = img.convert("RGB") 
            
            img.save(output_path, "JPEG")
            print(f"Image {image_path} converted to jpeg")

    except Exception as e:
        print(f"Failed to convert image: {e}")


if __name__ == "__main__":
    os.chdir('nn_files/photos/')
    dirs = get_list_dir()

    for each_dir in dirs:
        os.chdir(each_dir)
        files = get_list_dir()
        for each_file in files:
            if imghdr.what(each_file) != 'jpeg':
                convert_to_jpeg(each_file)

                os.remove(each_file)
                print(each_file + ' was removed!')
        os.chdir('../')