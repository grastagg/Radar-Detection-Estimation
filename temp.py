import os
import matplotlib.pyplot as plt


folder = "./saved_data/ratioData/expCov1/expDist10"
print("Files in", folder)
for dir in os.listdir(folder):
    print(dir)

    file = folder + "/" + dir + "/optimization/high_priority_path/noPathFound.png"
    if os.path.isfile(file):
        img = plt.imread(
            folder + "/" + dir + "/optimization/high_priority_path/noPathFound.png"
        )
        plt.imshow(img)
        plt.show()
    elif os.path.isfile(
        folder + "/" + dir + "/optimization/high_priority_path/spline.png"
    ):
        file = folder + "/" + dir + "/optimization/high_priority_path/spline.png"
        img = plt.imread(file)
        plt.imshow(img)
        plt.show()
    else:
        print("neither found")
