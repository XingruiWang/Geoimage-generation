import os

# val.lst
dir = "/home/hanfang_yang/GIS/data/geoimage/hold/images"

l = os.listdir(dir)
l.sort()
f = open("dataset/list/geoimage/val-mask.lst", "w")
fore_name = ""
for name in l:
    image_type = name.split("_")[2]
    to_write = "geoimage/hold/images/"+fore_name + "\t" + "geoimage/hold/images/"+name + "\n"
    if image_type == "pre":
        f.write(to_write)
    fore_name = name

# train.lst
dir = "/home/hanfang_yang/GIS/data/geoimage/train/images"

l = os.listdir(dir)
l.sort()
f = open("dataset/list/geoimage/train-mask.lst", "w")
fore_name = ""
for name in l:
    image_type = name.split("_")[2]
    to_write = "geoimage/train/images/"+fore_name + "\t" + "geoimage/train/images/"+name + "\n"
    if image_type == "pre":
        f.write(to_write)
    fore_name = name
