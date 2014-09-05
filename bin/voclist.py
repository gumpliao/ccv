#!/usr/bin/env python

from __future__ import print_function
import sys
import os

"""Lists images containing positive instances of a target class."""

__author__ = "Michele Lo Russo < michele.lo-russo13 @ imperial.ac.uk >"

def printusage(out=True, err=False):
        if out and not err:
            stream=sys.stdout
        else:
            stream=sys.stderr
        print("Usage:\n\n\tvoclist.py [annotations folder] [target class]\n", file=stream)
        print("\t[annotations folder] = folder containing VOC devkit annotation (.xml) files", file=stream)
        print("\t[target class] = name of target class as in the VOC dataset (e.g. person, bicycle)\n", file=stream)

def main():
    argn = len(sys.argv)
    if argn > 1 and ('--help' in sys.argv or '-h' in sys.argv or '-H' in sys.argv):
        print("")
        printusage()
        return
    if argn != 3:
        print("\nERROR: incorrect number of input parameters.\n", file=sys.stderr)
        printusage(out=False, err=True)
        sys.exit(0)
        
    folder = sys.argv[1]
    target = sys.argv[2].lower()
    target_tag = "<name>"+target+"</name>"
    if not folder.endswith('/'):
        folder += '/'
        
    print("\nReading annotation files...")
    filelist = [folder+f for f in os.listdir(folder) if os.path.isfile(folder+f) and f.endswith(".xml")]
    for i in range(len(filelist)):
        filelist[i] = open(filelist[i], 'r').read()

    print("Listing positives...")
    img_folder = folder[:folder[:-1].rindex("/")]+"/JPEGImages/"
    start = "<filename>"
    end = "</filename>"
    posex = [img_folder+f[(f.index(start)+10):f.index(end)] for f in filelist if target_tag in f]
    
    print("Saving file...")
    filelist = target+"_filelist.txt"
    with open(filelist, 'w'):
        for filename in posex:
            print(filename, file=open(filelist, 'a'))
    print("Done!\n")

if __name__ == '__main__':
    main()
