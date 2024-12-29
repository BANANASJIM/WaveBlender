# (c) 2024 Kangrui Xue
#
# run_all.py
# Script for running WaveBlender on all Scenes (must be executed from build/ directory)

import sys, os, subprocess
import numpy as np
import json


def run_all(baseDir):
    """
        baseDir: path to Scenes directory
    """
    if not os.path.exists("RUN_ALL/"):
        os.makedirs("RUN_ALL")

    # Loop through all subdirectories of Scenes/ base directory 
    for scene in os.listdir(baseDir):

        if not os.path.isdir(baseDir + "/" + scene):  # Skip if not a directory
            continue

        print("Running:", scene)
        configFile = baseDir + scene + "/config.json"
        subprocess.run(["./Release/WaveBlender.exe", configFile])

        try:
            with open(configFile) as f:
                config = json.load(f)
                
                samplerate = config["FDTD_srate"]
                output_name = config["listeners"]["1. Main Listener"]["output"]

                subprocess.run(["python", "../scripts/write_wav.py", output_name + ".bin", str(samplerate)])
                os.replace(output_name + ".wav", "RUN_ALL/" + output_name + ".wav")

        except:
            print("*** ERROR ***")


if __name__ == "__main__":
    baseDir = "../Scenes/" if (len(sys.argv) == 1) else sys.argv[1]
    run_all(baseDir)