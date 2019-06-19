#!/usr/bin/env python2
"""
Split out a measurement set for a given target.

NOTE: Use `execfile` in casa to use this file
"""
from __future__ import (print_function, division)
import os
import glob
import shutil


FIELDS = [
    'G2432',
    'G2984',
    'G3302',
    'G3604',
    'G4029',
]


def new_dirs():
    """
    Create new directories for the measurement sets.
    """
    for target in FIELDS:
        if not os.path.exists(target):
            print(':: Create folder')
            os.makedirs(target)


def split_calibrated(overwrite=False):
    """
    Split off the science targets and the science spectral windows.
    """
    # match full ms, not `target.ms`
    vislist = glob.glob('uid*[!_t].ms')
    print(':: Split science SPWs')
    for vis in vislist:
        print('-- {0}'.format(vis))
        msmd.open(vis)
        target_spws = msmd.spwsforintent('OBSERVE_TARGET*')  
        science_spws = [spw for spw in target_spws if msmd.nchan(spw) > 4]
        science_spws = ','.join(map(str, science_spws))
        msmd.close()
        if overwrite:
            rmtables(vis+'.split.cal')
        split(vis=vis,
              outputvis=vis+'.split.cal',
              intent='*TARGET*',
              spw=science_spws,
              datacolumn='corrected')
    print(':: Concat into calibrated.ms')
    sc_vislist = [s + '.split.cal' for s in vislist]
    if len(sc_vislist) == 1:
        shutil.move(sc_vislist[0], 'calibrated.ms')
    else:
        concat(vis=sc_vislist, concatvis='calibrated.ms')
    print(':: Creating listobs file')
    listobs(vis='calibrated.ms',
            listfile='calibrated.listobs.txt',
            overwrite=True)


def write_listobs(target):
    base = '{0}/calibrated_{0}'.format(target)
    for ext in ['', '_7m', '_joint']:
        filen = base + ext
        listobs(vis='{0}.ms'.format(filen),
                listfile='{0}.listobs.txt'.format(filen),
                overwrite=True)


