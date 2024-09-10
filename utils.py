import numpy as np
import os

def create_test_points(numTestPoints, bounds):
    x_test = np.linspace(0,bounds[0],numTestPoints)
    y_test = np.linspace(0,bounds[1],numTestPoints)

    X_test = []
    
    for i in range(numTestPoints):
        for j in range(numTestPoints):
            X_test.append(np.array([x_test[i],y_test[j]]))
    return np.array(X_test)
    

def change_random_seed(file_path, new_seed):
    # Read the existing script
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find and replace the line with `randomSeed`
    for i, line in enumerate(lines):
        if line.strip().startswith("randomSeed ="):
            lines[i] = f"randomSeed = {new_seed}\n"
            break
    
    # Write the updated script back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print(f"randomSeed changed to {new_seed} in {file_path}")  

def copy_params(dataFile):
    os.system("cp params.py "+dataFile +"params.py ")

def create_data_file(filepath,numRadar):
    print("creating data file: ",filepath)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
        # os.mkdir(filepath+"estimated_params/")
        # os.mkdir(filepath+"estimated_params_cov/")
        os.mkdir(filepath+"/high_priority_path/")
        for i in range(numRadar):
            os.mkdir(filepath+"radar_"+str(i)+"/")
            os.mkdir(filepath+"radar_"+str(i)+"/estimated_params/")
            os.mkdir(filepath+"radar_"+str(i)+"/estimated_params_cov/")
    else:
        print("Directory already exists")
        return
        # cont = input("overwrite? y/n")
        # if cont == "y":
        #     return
        # else:
        #     exit()