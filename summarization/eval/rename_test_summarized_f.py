import os

datapath = os.path.join(os.getcwd(),"test")
for filename in os.listdir(datapath):
    file_loc_name = os.path.join(datapath, filename)
    os.rename(os.path.join(datapath, filename), os.path.join(datapath, filename.replace(".html",".txt") ))