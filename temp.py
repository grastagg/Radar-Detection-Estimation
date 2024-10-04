import os


def move_files_excluding_lawnmower(parent_directory):
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        if os.path.isdir(item_path):
            new_folder_path = os.path.join(item_path, 'optimization')
            os.makedirs(new_folder_path, exist_ok=True)
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                if sub_item != 'lawnmower' and sub_item != 'optimization':
                    os.rename(sub_item_path, os.path.join(new_folder_path, sub_item))


move_files_excluding_lawnmower("./saved_data/mc_runs/")
