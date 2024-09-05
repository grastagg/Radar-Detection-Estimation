import numpy as np

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