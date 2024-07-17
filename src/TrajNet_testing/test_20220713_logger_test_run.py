import runpy
import utils
import os

subjs = utils.ExpInfo.getSubjIDs()[0:2]
for subj in subjs:
    gDict = {}
    gDict['subj'] = subj
    os.system(f"python '/home/acercyc/projects/Keio Testing_analysis/src/test_20220713_logger_test.py' {subj}")

    # runpy.run_path('/home/acercyc/projects/Keio Testing_analysis/src/test_20220713_logger_test.py', init_globals=gDict, run_name="__main__")
    
