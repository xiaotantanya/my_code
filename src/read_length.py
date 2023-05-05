from os import listdir
from os.path import isfile, join
from natsort import natsorted
import xml.etree.ElementTree as ET


file_list = listdir("./hievents_v2")
print(file_list)

all_files = natsorted([f for f in listdir("./hievents_v2") if isfile(join("./hievents_v2",f)) and f[-3:]=="xml"])
print(all_files)
max = 0

for file in all_files:
    tree = ET.parse("./hievents_v2/"+file)
    root = tree.getroot()
    doc_content=root[0].text
    length = len(doc_content.split(" "))
    if length>=max:
        max=length

print(max)