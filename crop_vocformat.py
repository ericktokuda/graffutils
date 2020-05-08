"""
Crops the images in subfolders of current directory according to annotations in 'Annotation' Folder and places then in 'Cropped'
Instructions: 
    1.In an empty directory, store this script and all the folders containing the images that will be processed
    2.In the same directory, add the \Annotation\ folder, containing a respective folder for each one in the directory, which in turn contain the annotation files
    3.run python cropImages.py
    4.Output will be in \Cropped\
"""
import os
from PIL import Image
import xml.etree.ElementTree as ET
import sys


annotdir = 'xml'
imdir = 'img'
outdir = '/tmp'

for filename in os.listdir(annotdir):
    rects = []
    basename =  os.path.splitext(filename)[0]
    print(filename)

    root = ET.parse(os.path.join(annotdir, filename))
    filename = root.find('filename').text
    objects = root.findall('object')

    img = Image.open(os.path.join(imdir, basename + '.jpg'))
    i = 0
    for object_iter in objects:
        bndbox = object_iter.find("bndbox")
        bboxcoords = [int(it.text) for it in bndbox]
        rects.append(bboxcoords)
        cropped = img.crop((bboxcoords[0], bboxcoords[1], bboxcoords[2], bboxcoords[3]))
        outcropname = '{}_{}.jpg'.format(basename, i)
        cropped.save(os.path.join(outdir, outcropname), "JPEG")
        i += 1

    print(len(rects))
