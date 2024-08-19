import os

def scan_directory_for_jpgs(directory_path, output_file):
    with open(output_file, 'w') as out_file:
        # Iterate over all files in the directory
        for root, dirs, files in os.walk(directory_path):
            for file_name in files:
                # Check if the file has a .jpg extension
                if file_name.lower().endswith('.jpg'):
                    file_path = os.path.join(root, file_name)
                    out_file.write(f'build/darknet/x64/data/obj/{file_name}\n')

if __name__ == "__main__":
    directory_path = '/Users/jan/Documents/code/cv/project/darknet/build/darknet/x64/data/obj'  # Specify the directory you want to scan
    output_file = 'train.txt'  # Name of the output file
    
    scan_directory_for_jpgs(directory_path, output_file)
