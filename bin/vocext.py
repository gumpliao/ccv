#!/usr/bin/env python

from __future__ import print_function
import sys
import os

"""Produces two lists of image files, for positive and negative examples of the given target class."""

__author__ = "Michele Lo Russo < michele.lo-russo13 @ imperial.ac.uk >"

def printusage(out=True, err=False):
        if out and not err:
            stream=sys.stdout
        else:
            stream=sys.stderr
        print("Usage:\n\n\tvocext.py [annotations folder] [target class] ...\n", file=stream)
        print("\t[annotations folder] = folder containing VOC devkit annotation (.xml) files", file=stream)
        print("\t[target class] = name of target class as in the VOC dataset (e.g. person, bicycle)\n", file=stream)
        print("\tSPECIAL KEYWORDS: ALL produces files for all the 20 classes, REMAINDER only for\n\tthe classes with no files in <annotationsfolder> (assuming <targetclass>.samples\n\tand no-<targetclass>.samples format).\n\tBoth modes force <targetclass>.samples and no-<targetclass>.samples format for\n\tfilenames, and optional parameters are ignored.\n", file=stream)
        print("\t... = optional parameters:", file=stream)
        print("\t[posex list] = where to save list of positive examples", file=stream)
        print("\t[negex list] = where to save list of negative examples\n", file=stream)
        print("\tIf no optional parameters are specified, by default the lists are saved in files\n\t<targetclass>.samples and no-<targetclass>.samples in <annotationsfolder>\n\t(where <targetclass> is the name as in the VOC dataset).\n", file=stream)
        print("\tWARNING: If the output files already exist, their content will be overwritten.\n", file=stream)

def getremainder(filelist):
    targetlist = []
    if 'aeroplane.samples' not in filelist or 'no-aeroplane.samples' not in filelist:
        targetlist.append('aeroplane')
    if 'bicycle.samples' not in filelist or 'no-bicycle.samples' not in filelist:
        targetlist.append('bicycle')
    if 'bird.samples' not in filelist or 'no-bird.samples' not in filelist:
        targetlist.append('bird')
    if 'boat.samples' not in filelist or 'no-boat.samples' not in filelist:
        targetlist.append('boat')
    if 'bottle.samples' not in filelist or 'no-bottle.samples' not in filelist:
        targetlist.append('bottle')
    if 'bus.samples' not in filelist or 'no-bus.samples' not in filelist:
        targetlist.append('bus')
    if 'car.samples' not in filelist or 'no-car.samples' not in filelist:
        targetlist.append('car')
    if 'cat.samples' not in filelist or 'no-cat.samples' not in filelist:
        targetlist.append('cat')
    if 'chair.samples' not in filelist or 'no-chair.samples' not in filelist:
        targetlist.append('chair')
    if 'cow.samples' not in filelist or 'no-cow.samples' not in filelist:
        targetlist.append('cow')
    if 'diningtable.samples' not in filelist or 'no-diningtable.samples' not in filelist:
        targetlist.append('diningtable')
    if 'dog.samples' not in filelist or 'no-dog.samples' not in filelist:
        targetlist.append('dog')
    if 'horse.samples' not in filelist or 'no-horse.samples' not in filelist:
        targetlist.append('horse')
    if 'motorbike.samples' not in filelist or 'no-motorbike.samples' not in filelist:
        targetlist.append('motorbike')
    if 'person.samples' not in filelist or 'no-person.samples' not in filelist:
        targetlist.append('person')
    if 'pottedplant.samples' not in filelist or 'no-pottedplant.samples' not in filelist:
        targetlist.append('pottedplant')
    if 'sheep.samples' not in filelist or 'no-sheep.samples' not in filelist:
        targetlist.append('sheep')
    if 'sofa.samples' not in filelist or 'no-sofa.samples' not in filelist:
        targetlist.append('sofa')
    if 'train.samples' not in filelist or 'no-train.samples' not in filelist:
        targetlist.append('train')
    if 'tvmonitor.samples' not in filelist or 'no-tvmonitor.samples' not in filelist:
        targetlist.append('tvmonitor')
    return targetlist

def checktarget(target):
    return target == 'aeroplane' or target == 'bicycle' or target == 'bird' or target == 'boat' or target == 'bottle' or target == 'bus' or target == 'car' or target == 'cat' or target == 'chair' or target == 'cow' or target == 'diningtable' or target == 'dog' or target == 'horse' or target == 'motorbike' or target == 'person' or target == 'pottedplant' or target == 'sheep' or target == 'sofa' or target == 'train' or target == 'tvmonitor'

def getbndboxdata(f, target_tag, n_obj):
    targets, boxdata = [], []
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
        boxdata.append(" "+str(int(round(float(xmin))))+" "+str(int(round(float(ymin))))+" "+str(int(round(float(xmax)-float(xmin))))+" "+str(int(round(float(ymax)-float(ymin)))))
    return boxdata

def findposneg(target, filelist):
    posex = []
    negex = []
    start = "<filename>"
    end = "</filename>"
    target_tag = "<name>"+target+"</name>"
    for f in filelist:
        filename = f[(f.index(start)+10):f.index(end)]
        if target_tag in f:
            n_obj = f.count(target_tag)
            boxdata = getbndboxdata(f,target_tag,n_obj)
            for i in range(n_obj):
                posex.append(filename+boxdata[i])
        else:
            negex.append(filename)

    return posex, negex

def main():
    argn = len(sys.argv)
    if argn > 1 and ('--help' in sys.argv or '-h' in sys.argv or '-H' in sys.argv):
        print("")
        printusage()
        return
    if argn < 3:
        print("\nERROR: incorrect number of input parameters.\n", file=sys.stderr)
        printusage(out=False, err=True)
        sys.exit(0)

    folder = sys.argv[1]
    target = sys.argv[2].lower()
    if not folder.endswith('/'):
        folder += '/'
    if target == 'all':
        targetlist = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif target == 'remainder':
        filelist = [f for f in os.listdir(folder) if os.path.isfile(folder+f) and f.endswith(".samples")]
        targetlist = getremainder(filelist)
        if targetlist == []:
            return
    else:
        if not checktarget(target):
            print("\nERROR:\n\tUnrecognised class: "+target+"\n")
            sys.exit(-1)
        targetlist = [target]

    print("\nReading annotation files...")
    filelist = [folder+f for f in os.listdir(folder) if os.path.isfile(folder+f) and f.endswith(".xml")]
    for i in range(len(filelist)):
        filelist[i] = open(filelist[i], 'r').read()

    print("Finding positives and negatives...")
    for target in targetlist:
        posex, negex = findposneg(target, filelist)
        if argn < 4 or target == 'all' or target == 'remainder':
            posout = folder+target+".samples"
        else:
            posout = sys.argv[3]
        with open(posout, 'w'):
            for filename in posex:
                print(filename, file=open(posout, 'a'))
        if argn < 5 or target == 'all' or target == 'remainder':
            negout = folder+"no-"+target+".samples"
        else:
            negout = sys.argv[4]
        with open(negout, 'w'):
            for filename in negex:
                print(filename, file=open(negout, 'a'))
    print("Done!\n")

if __name__ == '__main__':
    main()
