import os



def check_mc_runs(runDirectory):
    count = 0
    for temp in os.listdir(runDirectory):
        print(temp)
        if os.path.isdir(runDirectory + "/" + temp):
            tempDir = runDirectory + "/" + temp + "/optimization/high_priority_path/controlPoints.txt"
            if os.path.isfile(tempDir):
                # print("File exists")
                count += 1
                # tempDir2 = runDirectory + "/" + temp
                # print("Directory: ", tempDir2)
                # os.rename(tempDir2, runDirectory + "/tempOpt/" +temp)
            # else:
            #     print("File does not exist")
                # os.rmdir(runDirectory + "/" + temp)
    print("Total number of files: ", count)
        

if __name__ == "__main__":
    check_mc_runs("./saved_data/mc_runs")