import imghdr
import os

def get_list_dir():
    list_dir = os.listdir()
    list_dir.remove('.DS_Store')
    return list_dir

os.chdir('nn_files/photos/')
dirs = get_list_dir()

for each_dir in dirs:
    os.chdir(each_dir)
    files = get_list_dir()
    for each_file in files:
        if imghdr.what(each_file) != 'jpeg':
            os.remove(each_file)
            print(each_file + ' was removed!')
    os.chdir('../')
