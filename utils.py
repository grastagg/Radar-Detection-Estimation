import numpy as np
import os


def create_test_points(numTestPoints, bounds):
    x_test = np.linspace(0, bounds[0], numTestPoints)
    y_test = np.linspace(0, bounds[1], numTestPoints)

    X_test = []

    for i in range(numTestPoints):
        for j in range(numTestPoints):
            X_test.append(np.array([x_test[i], y_test[j]]))
    return np.array(X_test)


def change_random_seed(file_path, new_seed):
    # Read the existing script
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find and replace the line with `randomSeed`
    for i, line in enumerate(lines):
        if line.strip().startswith("randomSeed ="):
            lines[i] = f"randomSeed = {new_seed}\n"
            break

    # Write the updated script back to the file
    with open(file_path, "w") as file:
        file.writelines(lines)

    print("HERE")
    print(f"randomSeed changed to {new_seed} in {file_path}")


def change_parameter_ratios(file_path, expCovRatio, expDistRatio):
    # Read the existing script
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find and replace the line with `randomSeed`
    for i, line in enumerate(lines):
        if line.strip().startswith("sepUncertaintyRatio ="):
            lines[i] = f"sepUncertaintyRatio = {expCovRatio}\n"
        if line.strip().startswith("sepDistRatio ="):
            lines[i] = f"sepDistRatio = {expDistRatio}\n"
            break

    # Write the updated script back to the file
    with open(file_path, "w") as file:
        file.writelines(lines)

    print(
        f"expCovRatio changed to {expCovRatio} and expDistRatio changed to {expDistRatio} in {file_path}"
    )


def change_weights(file_path, explorationWeight, covarianceWeight, distWeight):
    # Read the existing script
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find and replace the line with `randomSeed`
    for i, line in enumerate(lines):
        if line.strip().startswith("seperationWeight ="):
            lines[i] = f"seperationWeight = {explorationWeight}\n"
        if line.strip().startswith("distFromStraitWeight ="):
            lines[i] = f"distFromStraitWeight = {distWeight}\n"
        if line.strip().startswith("nextCovarianceWeight ="):
            lines[i] = f"nextCovarianceWeight = {covarianceWeight}\n"
            break

    # Write the updated script back to the file
    with open(file_path, "w") as file:
        file.writelines(lines)

    print(
        f"explorationWeight changed to{explorationWeight}, nextCovarianceWeight changed to {covarianceWeight}, distWeight changed to {distWeight} "
    )


def change_num_agents(file_path, numAgents):
    with open(file_path, "r") as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("numAgents ="):
            print(lines[i])
            lines[i] = f"numAgents = {numAgents}\n"
            print("numAgents changed to ", numAgents)
            break
    with open(file_path, "w") as file:
        file.writelines(lines)


def set_path_planner(file_path, pathPlanner):
    # Read the existing script
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find and replace the line with `randomSeed`
    for i, line in enumerate(lines):
        if line.strip().startswith("lowPriorityPathPlanner ="):
            lines[i] = f'lowPriorityPathPlanner = "{pathPlanner}"\n'
            break

    # Write the updated script back to the file
    with open(file_path, "w") as file:
        file.writelines(lines)

    print(f"pathPlanner changed to {pathPlanner} in {file_path}")


def copy_params(dataFile):
    os.system("cp params.py " + dataFile + "params.py ")


def create_data_file(filepath, numRadar, pathPlanner):
    print("creating data file: ", filepath)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    print("creating subdirectories")
    if not os.path.exists(filepath + "/" + pathPlanner + "/"):
        filepath = filepath + "/" + pathPlanner + "/"
        os.mkdir(filepath)
        # os.mkdir(filepath+"estimated_params/")
        # os.mkdir(filepath+"estimated_params_cov/")
        os.mkdir(filepath + "/high_priority_path/")
        for i in range(numRadar):
            os.mkdir(filepath + "radar_" + str(i) + "/")
            # os.mkdir(filepath + "radar_" + str(i) + "/estimated_params/")
            # os.mkdir(filepath + "radar_" + str(i) + "/estimated_params_cov/")
        # cont = input("overwrite? y/n")
        # if cont == "y":
        #     return
        # else:
        #     exit()
