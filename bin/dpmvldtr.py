#!/usr/bin/env python

import sys
import os
import re

"""Direct translation of the original dpmvldtr script from RUBY to PYTHON."""

__author__ = "Michele Lo Russo < michele.lo-russo13 @ imperial.ac.uk >"

def getbndboxdata(f):
    targets, boxes = [], []
    for line in f.splitlines():
        if line[0:16].lower() == "bounding box for":
            coords = re.findall(r"[\w']+", line[line.rindex(" : ")+3:])
            boxdata = {}
            boxdata['found'] = False
            boxdata['x'] = int(round(float(coords[0])))
            boxdata['y'] = int(round(float(coords[1])))
            boxdata['width'] = int(round(float(coords[2])-float(coords[0])))
            boxdata['height'] = int(round(float(coords[3])-float(coords[1])))
            boxes.append(boxdata)
    return boxes

def main():
    argn = len(sys.argv)
    if argn != 3:
        sys.exit(-1)
        
    folder = sys.argv[1]
    if not folder.endswith('/'):
        folder += '/'
    
    truth, total = {}, 0
    
    filelist = [folder+f for f in os.listdir(folder)]
    for i in range(len(filelist)):
        filelist[i] = open(filelist[i], 'r').read()
    for f in filelist:
        start = "Image filename : "
        end = "\"\n"
        name = f[(f.index(start)+17):f.index(end)]
        name = name[name.rindex("/")+1:]
        boxes = getbndboxdata(f)
        truth[name] = boxes
        total += len(boxes)
        
    fa, tp = 0, 0
    
    results = open(sys.argv[2], 'r')
    line = results.readline()
    while line:
        if line[0] == '|':
            line = results.readline()
            continue
        args = line.split(" ")
        name = args[0][args[0].rindex("/")+1:]
        if name not in truth:
            fa += 1
        else:
            x = int(args[1])
            y = int(args[2])
            width = int(args[3])
            height = int(args[4])
            outlier = -1
            for obj in truth[name]:
                opx_min = max(obj['x'], x)
                opy_min = max(obj['y'], y)
                opx_max = min(obj['x'] + obj['width'], x + width)
                opy_max = min(obj['y'] + obj['height'], y + height)
                r0 = max(opx_max - opx_min, 0) * max(opy_max - opy_min, 0)
                r1 = max(obj['width'] * obj['height'], width * height) * 0.5
                if r0 > r1:
                    outlier = 0 if obj['found'] else 1
                    obj['found'] = True
                    break
            if outlier == -1:
                fa += 1
            elif outlier == 1:
                tp += 1
        line = results.readline()
        
    print str(round(float(tp) / float(total)*10000) / 100.0) + "% (" + str(fa) + ")\n"

if __name__ == '__main__':
    main()
