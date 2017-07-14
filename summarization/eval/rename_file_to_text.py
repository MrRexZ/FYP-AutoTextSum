import os
from bs4 import BeautifulSoup

datapath = os.path.join(os.getcwd(),"input")
writetopath = os.path.join(os.getcwd(),"datatest")
for outer_filename in os.listdir(datapath):
    for filename in os.listdir(os.path.join(datapath,outer_filename)):
        file_loc_name = os.path.join(datapath,outer_filename,filename)
        with open(file_loc_name , "r+") as html_doc:
            soup = BeautifulSoup(html_doc, 'html.parser')
            html_doc.seek(0)
            html_doc.write(soup.find("text").get_text())
            html_doc.truncate()
        os.rename(file_loc_name, file_loc_name + ".txt")
