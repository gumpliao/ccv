#!/usr/bin/env python

from __future__ import print_function
import sys
import os

"""Alternative version of the original dpmvldtr script to validate detection on VOC datasets using CCV metrics."""

__author__ = "Michele Lo Russo < michele.lo-russo13 @ imperial.ac.uk >"

def printusage(out=True, err=False):
        if out and not err:
            stream=sys.stdout
        else:
            stream=sys.stderr
        print("Usage:\n\n\tvocvldtr.py [annotations folder] [target class] [results file]\n", file=stream)
        print("\t[annotations folder] = folder containing VOC devkit annotation (.xml) files", file=stream)
        print("\t[target class] = name of target class as in the VOC dataset (e.g. person, bicycle)", file=stream)
        print("\t[results file] = file containing detection data as printed to the standard output by dpmdetect or dpmsparsedetect\n", file=stream)

def getbndboxdata(f, target_tag, n_obj):
    targets, boxes = [], []
    while n_obj > 0:
        obj = f[f.index("<object>"):f.index("</object>")]
        if target_tag not in obj:
            f=f[(f.index("</object>")+9):]
            continue
        bndbox = obj[obj.index("<bndbox>"):obj.index("</bndbox>")]
        targets.append(bndbox)
        f=f[(f.index("</object>")+9):]
        n_obj = n_obj-1
    for box in targets:
        xmax = box[(box.index("<xmax>")+6):box.index("</xmax>")]
        xmin = box[(box.index("<xmin>")+6):box.index("</xmin>")]
        ymax = box[(box.index("<ymax>")+6):box.index("</ymax>")]
        ymin = box[(box.index("<ymin>")+6):box.index("</ymin>")]
        boxdata = {}
        boxdata['found'] = False
        boxdata['x'] = int(round(float(xmin)))
        boxdata['y'] = int(round(float(ymin)))
        boxdata['width'] = int(round(float(xmax)-float(xmin)))
        boxdata['height'] = int(round(float(ymax)-float(ymin)))
        boxes.append(boxdata)
    return boxes

def main():
    argn = len(sys.argv)
    if argn > 1 and ('--help' in sys.argv or '-h' in sys.argv or '-H' in sys.argv):
        print("")
        printusage()
        return
    if argn != 4:
        print("\nERROR: incorrect number of input parameters.\n", file=sys.stderr)
        printusage(out=False, err=True)
        sys.exit(0)
        
    folder = sys.argv[1]
    target = sys.argv[2].lower()
    target_tag = "<name>"+target+"</name>"
    if not folder.endswith('/'):
        folder += '/'
    
    truth, total = {}, 0
    
    print("\nReading annotation files...")
    filelist = [folder+f for f in os.listdir(folder) if os.path.isfile(folder+f) and f.endswith(".xml")]
    for i in range(len(filelist)):
        filelist[i] = open(filelist[i], 'r').read()

    print("Gathering bounding box data of positive instances...")
    for f in filelist:
        if target_tag not in f:
            continue
        start = "<filename>"
        end = "</filename>"
        name = f[(f.index(start)+10):f.index(end)]
        n_obj = f.count(target_tag)
        boxes = getbndboxdata(f, target_tag, n_obj)
        truth[name] = boxes
        total += len(boxes)
        
    fa, tp = 0, 0

    print("Computing recall and number of false positives...")
    results = open(sys.argv[3], 'r')
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

    print("Done!\n")
    print("RECALL: "+str(round(float(tp) / float(total)*10000) / 100.0) + "%")
    print("FALSE POSITIVES: " + str(fa) + "\n")

if __name__ == '__main__':
    main()
