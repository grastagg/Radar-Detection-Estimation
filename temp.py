import os
import matplotlib.pyplot as plt


def test_folder(folder):
    print("Files in", folder)
    for dir in os.listdir(folder):
        file = folder + "/" + dir + "/optimization/high_priority_path/noPathFound.png"
        if os.path.isfile(file):
            print("noPathFound")
            print(dir)
            img = plt.imread(
                folder + "/" + dir + "/optimization/high_priority_path/noPathFound.png"
            )
            plt.imshow(img)
            plt.show()
        elif os.path.isfile(
            folder + "/" + dir + "/optimization/high_priority_path/spline.png"
        ):
            pass
            # file = folder + "/" + dir + "/optimization/high_priority_path/spline.png"
            # img = plt.imread(file)
            # plt.imshow(img)
            # plt.show()
        else:
            print("neither found")


if __name__ == "__main__":
    folder = "./saved_data/ratioData/expCov4/expDist1"
    test_folder(folder)
    folder = "./saved_data/ratioData/expCov4/expDist2"
    test_folder(folder)
    folder = "./saved_data/ratioData/expCov4/expDist3"
    test_folder(folder)
    folder = "./saved_data/ratioData/expCov4/expDist4"
    test_folder(folder)
    folder = "./saved_data/ratioData/expCov4/expDist5"
    test_folder(folder)
    folder = "./saved_data/ratioData/expCov4/expDist6"
    test_folder(folder)
    folder = "./saved_data/ratioData/expCov4/expDist7"
    test_folder(folder)
    folder = "./saved_data/ratioData/expCov4/expDist8"
    test_folder(folder)
    folder = "./saved_data/ratioData/expCov4/expDist9"
    test_folder(folder)
    folder = "./saved_data/ratioData/expCov4/expDist10"
    test_folder(folder)
