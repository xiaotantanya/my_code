import xml.etree.ElementTree as ET

tree = ET.parse("./data_example/article-1126.xml")
root = tree.getroot()
print(root.attrib)