#!/bin/zsh

# Directory containing the folders
directory="./saved_data/test_with_new_seeds/"

# Path to the Python script
python_script="highPriorityPathPlanner.py"

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Error: Directory $directory does not exist."
  exit 1
fi

# Iterate through all folders in the directory
for folder in "$directory"/*/; do
  # Remove trailing slash from folder path
  folder=${folder%/}

  # Extract folder name
  folder_name=$(basename "$folder")

  # Run the Python script with the folder name as an argument
  echo "Processing folder: $folder_name"
  # python3 "$python_script" "$folder_name"

  # pathPlanner='lawnmower'
  # nohup python3 -u "$python_script" "$folder_name" "$pathPlanner" >outputsMC/lawnmower/$folder_name.log 2>&1 &
  pathPlanner='optimization'
  # nohup python3 -u "$python_script" "$folder_name" "$pathPlanner" >outputsMC/optimization/$folder_name.log 2>&1 &
  nohup python3 -u "$python_script" "$folder_name" "$pathPlanner" >outputsMC/optimization/$folder_name.log 2>&1 &

  echo "------------------------"
done
