import os
import shutil

# Define the main folder (current directory)
main_folder = "Dataset/"

# Iterate through all subfolders
for root, dirs, files in os.walk(main_folder):
    if root == main_folder:
        continue  # Skip the main folder itself
    
    for file in files:
        file_path = os.path.join(root, file)
        new_path = os.path.join(main_folder, file)
        
        # Ensure there are no name conflicts
        counter = 1
        while os.path.exists(new_path):
            file_name, file_ext = os.path.splitext(file)
            new_path = os.path.join(main_folder, f"{file_name}_{counter}{file_ext}")
            counter += 1
        
        shutil.move(file_path, new_path)

print("All files have been moved to the current folder.")