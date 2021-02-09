#!/usr/bin/env python
"""
============
Image Target
============
Create continuum and spectral line images for the ALMA targets.

NOTE: Run with `execfile` in CASA to use this script.
"""
from __future__ import (print_function, division)

import os
import shutil
import datetime
from glob import glob
from collections import namedtuple

import numpy as np


Spw = namedtuple('Spw', 'name, spw_id, restfreq, velo_width, nchan, ot_name')
SPWS = { spw.name : spw for spw in [
    Spw('h2o_6_5',    5, '22.2350798GHz', '0.843km/s', 240, 'EVLA_K#B0D0#5'),
    Spw('CCS_2_1',    6, '22.3440308GHz', '0.419km/s', 100, 'EVLA_K#B0D0#6'),
    Spw('hc7n_20_19', 7, '22.5599155GHz', '0.415km/s', 100, 'EVLA_K#B0D0#7'),
    Spw('hc7n_21_20', 0, '23.6878974GHz', '0.158km/s', 126, 'EVLA_K#A0C0#0'),
    Spw('nh3_11',     0, '23.6944955GHz', '0.158km/s', 380, 'EVLA_K#A0C0#0'),
    Spw('nh3_22',     1, '23.7226333GHz', '0.158km/s', 380, 'EVLA_K#A0C0#1'),
    Spw('nh3_33',     2, '23.8701292GHz', '0.393km/s', 100, 'EVLA_K#A0C0#2'),
    Spw('hc5n_9_8',   3, '23.9639007GHz', '0.391km/s', 100, 'EVLA_K#A0C0#3'),
    Spw('nh3_44',     4, '24.1394163GHz', '0.388km/s', 100, 'EVLA_K#A0C0#4'),
]}


Target = namedtuple('Target', 'name, field, vlsr')
TARGETS = { targ.name : targ for targ in [
    Target('G30912',  '2', 50.74),
    Target('G30660',  '3', 80.20),
    Target('G30120',  '4', 65.31),
    Target('G29601',  '5', 75.78),
    Target('G29558',  '6', 79.72),
    Target('G28539',  '7', 88.60),
    Target('G28565',  '8', 87.46),
    Target('G22695', '10', 77.80),
    Target('G23297', '11', 55.00),
    Target('G23481', '12', 63.80),
    Target('G23605', '13', 87.00),
    Target('G24051', '14', 81.51),
    Target('G285_mosaic', '7,8', 88.6),
]}


# Computed windows for:
# +/-  25 km/s for fixed doppler tracking at 75 km/s (LSR; sources 50-90 km/s)
# +/-  30 km/s for NH3 (1,1) & (2,2)
# +/- 100 km/s for H2O maser
# +/-  20 km/s for other lines
# NOTE the HC7N line in the NH3 (1,1) window shortens it and leaves no middle
BASELINE_CHANS = ','.join([
    '0:129~531;1628~2431',
    '1:129~932;1628~2341',
    '2:26~142;369~486',
    '3:26~142;369~486',
    '4:26~142;369~486',
    '5:26~106;405~486',
    '6:26~146;366~486',
    '7:26~142;366~486',
])


ROOT_DIR = '/lustre/aoc/users/bsvoboda/17A-146/data/'
IMG_DIR = ROOT_DIR + 'images/'
GBT_DIR = ROOT_DIR + 'gbt_cubes/'
CAT_DIR = ROOT_DIR + 'catalogs/'
VIS_FILES = sorted(glob('17A-146_*/pipe_specline/17A-146.sb*.ms'))
VIS_FILES = [ROOT_DIR + s for s in VIS_FILES]


###############################################################################
# General utility functions
###############################################################################

def log_post(msg):
    """
    Post a message to the CASA logger and logfile.
    """
    casalog.post(msg, 'INFO', 'bsvoboda')


def check_delete_image_files(imagename, parallel=False, preserve_mask=False):
    """
    Check for and remove (if they exist) files created by clean such as '.flux',
    '.image', etc.
    NOTE this function has issues with deleting tables made by clean in
    parallel mode, probably because the directory structure is different.

    Parameters
    ----------
    imagename : str
        The relative path name to the files to delete.
    parallel : bool, default False
        rmtables can't remove casa-images made with parallel=True, they must be
        manually removed.
    preserve_mask : bool, default False
        Whether to preserve the `.mask` file extension
    """
    log_post(':: Check for and remove existing files')
    exts = [
        '.flux', '.pb', '.image', '.weight', '.model', '.pbcor', '.psf',
        '.sumwt', '.residual', '.flux.pbcoverage',
    ]
    if not preserve_mask:
        exts += ['.mask']
    # CASA image table directories
    for ext in exts:
        filen = imagename + ext
        if os.path.exists(filen):
            if parallel:
                log_post('-- Hard delete {0}'.format(ext))
                shutil.rmtree(filen)
            else:
                log_post('-- Removing {0}'.format(filen))
                rmtables(filen)
    # "Cannot delete X because it's not a table" -> so hard delete
    for ext in ('.residual', '.workdirectory'):
        filen = imagename + ext
        if os.path.exists(filen):
            log_post('-- Hard delete {0}'.format(ext))
            shutil.rmtree(filen)


def export_fits(imagename, overwrite=True):
    log_post(':: Exporting fits')
    exportfits(imagename, imagename+'.fits', velocity=True, overwrite=overwrite)


def if_exists_remove(imagename):
    if os.path.exists(imagename):
        rmtables(imagename)


def export_fits_all(ext='image'):
    for imagename in glob(IMG_DIR + 'G*/*.{0}'.format(ext)):
        log_post(':: Export {0} to FITS'.format(imagename))
        export_fits(imagename)


def clean_workdirs_all():
    for targ in TARGETS.keys():
        for path in glob(IMG_DIR+'{0}/{0}_*.workdirectory'.format(targ)):
            log_post(':: Hard delete {0}'.format(path))
            shutil.rmtree(path)


def primary_beam_correct(imagename, overwrite=True, export=True):
    log_post(':: Primary beam correct {0}'.format(imagename))
    imagebase, ext = os.path.splitext(imagename)
    if imagebase.endswith('_jfeather'):
        pbimage = imagebase.replace('jfeather', 'joint') + '.pb'
    else:
        pbimage = imagebase + '.pb'
    pbcorimage = imagebase + '.pbcor'
    impbcor(imagename=imagename, pbimage=pbimage, outfile=pbcorimage,
            overwrite=overwrite)
    if export:
        export_fits(pbcorimage)


def primary_beam_correct_all(ext='image', export=True):
    for imagename in glob(IMG_DIR + 'G*/*.{0}'.format(ext)):
        primary_beam_correct(imagename)


def calc_start_velo(targ, spw):
    window = float(spw.velo_width.strip('km/s')) * spw.nchan
    start_velo = '{0:.4f}km/s'.format(targ.vlsr - window / 2)
    return start_velo


def concat_parallel_image(imagename, ext='image'):
    outfile = imagename + '.concat'
    if_exists_remove(outfile)
    ia.open(imagename)
    im_tool = ia.imageconcat(
            outfile=outfile,
            infiles=imagename+'/*.{0}'.format(ext),
            reorder=True,
    )
    im_tool.close()
    ia.done()


def convert_to_common_beam(imagename):
    outfile = imagename + '.combm'
    if_exists_remove(outfile)
    imsmooth(imagename, kernel='common', outfile=outfile)


###############################################################################
# Calibration
###############################################################################

def do_statwt_all():
    """
    The pipeline task `hifv_statwt` was removed from the script, so the
    measurement sets need to have their weight spectrums computed excluding the
    channels with line emission.

    NOTE: this must be run using CASA 5.4.1 (pre-release as of 10/23/18) in
    order to have the `fitspw` keyword argument.
    """
    log_post(':: Apply statwt to all measurement sets')
    for vis in VIS_FILES:
        log_post('-- {0}'.format(os.path.basename(vis)))
        statwt(vis=vis, fitspw=BASELINE_CHANS)


def split_test_data(one_eb=True, few_chans=False):
    """
    Split out a small test data set on one EB, one target G28539, and the NH3
    (1,1) line.

    Parameters
    ----------
    one_eb : bool
        Split just one execution block ms (the second one, a full block).
    few_chan : bool
        Split 63 chan (10 km/s) around the main NH3 (1,1) line.
    """
    targ = TARGETS['G28539']
    outputvis_base = ROOT_DIR + 'test_imaging/test_split'
    if few_chans:
        spw = '{0}:1159~1222'.format(SPWS['nh3_11'].spw_id)
    else:
        spw = '{0}'.format(SPWS['nh3_11'].spw_id)
    all_vis = [VIS_FILES[1]] if one_eb else VIS_FILES
    for i, vis in enumerate(all_vis):
        outputvis = '{0}_eb{1}.ms'.format(outputvis_base, i)
        split(
            vis=vis,
            outputvis=outputvis,
            field=targ.field,
            spw=spw,
        )


###############################################################################
# Line imaging
###############################################################################

def tune_automask_clean(imagename):
    """
    Test the results of the auto-multithresh masking method for different input
    parameters. A special dataset is created in "test_imaging/test_automask"
    for the NH3 (1,1) line with a few bright channels on a single SB to make it
    quick to process.
    """
    log_post(':: Running clean for {0}'.format(imagename))
    # "test_split.ms" contains only NH3 (1,1) spw
    spw = SPWS['nh3_11']
    imagename = ROOT_DIR + 'test_imaging/test_automask/{0}'.format(imagename)
    tclean(
        vis=ROOT_DIR+'test_imaging/test_split_1eb.ms',
        imagename=imagename,
        specmode='cube',
        outframe='lsrk',
        veltype='radio',
        restfreq=spw.restfreq,
        nchan=-1,
        imsize=[350, 350],  # 175 as, 350 "efficient size"
        cell='0.5arcsec',  # 3.4 as / 7
        # gridder parameters
        gridder='awproject',
        wprojplanes=1,
        aterm=True,
        rotatepastep=5.0,
        pblimit=1e-5,
        # deconvolver parameters
        deconvolver='multiscale',
        scales=[0, 7, 14, 35],  # point, 1, 3, 5 beam hpbw's
        smallscalebias=0.6,
        restoringbeam='common',
        weighting='briggs',
        robust=1.0,
        niter=1000000,
        threshold='5mJy',  # rms -> 5 mJy/beam
        cyclefactor=2,
        interactive=False,
        verbose=True,
        # automasking parameters
        usemask='auto-multithresh',  # use ALMA 12m(short) values
        noisethreshold=2.0,
        sidelobethreshold=1.0,
        lownoisethreshold=1.5,
        minbeamfrac=2.0,
        negativethreshold=1000.0,
    )


def test_clean_line_target(targ, spw, ext=None, startmodel=False, iterzero=False,
        restart=False):
    """
    Run tclean on a target. Uses multi-scale clean and automasking. Optional to
    use single-dish as a startmodel.

    Parameters
    ----------
    startmodel : bool, default False
        Use the GBT single-dish image as a starting model
    ext : string, default None
        Extension to add to imagename
        Example: 'joint' for 'G28539_nh3_11_joint' base
    iterzero : bool, default False
        Create a dirty map and other products by setting `niter=0`, otherwise
        `niter=1e6` .
    restart : bool, default False
    """
    log_post(':: Running clean ({0}, {1})'.format(targ.name, spw.name))
    start_velo = calc_start_velo(targ, spw)
    imagename = 'images/{0}/{0}_{1}'.format(targ.name, spw.name)
    if ext is not None:
        imagename = '{0}_{1}'.format(imagename, ext)
    if startmodel:
        startmodel = 'gbt_cubes/for_joint/{0}_{1}.jypix'.format(targ.name, spw.name)
    else:
        startmodel = None
    # restart parameters
    niter = 0 if iterzero else int(1e6)
    calcpsf = not restart  # ie True when not restarting
    calcres = not restart
    # scale typical RMS of 1.8 mJy/beam per 0.158 km/s channel for NH3 (1,1) to
    # other SPWs
    rms = 1.8 * np.sqrt(0.158 / float(spw.velo_width.strip('km/s')))
    threshold = '{0:.4f}mJy'.format(rms)
    imagename = 'test_imaging/test_mpicasa_bugfix/image'  # XXX
    check_delete_image_files(imagename, parallel=True)
    tclean(
        vis=VIS_FILES,
        imagename=imagename,
        field=targ.field,
        spw=str(spw.spw_id),
        specmode='cube',
        outframe='lsrk',
        veltype='radio',
        restfreq=spw.restfreq,
        start=start_velo,
        nchan=spw.nchan,
        imsize=[350, 350],  # 175 as, 350 "efficient size"
        cell='0.5arcsec',  # 3.4 as / 7
        # gridder parameters
        gridder='standard',
        # deconvolver parameters
        deconvolver='multiscale',
        scales=[0, 7, 21],  # point, 1, 3 beam hpbw's
        smallscalebias=0.6,
        restoringbeam='common',
        weighting='briggs',
        robust=1.0,
        niter=niter,
        threshold=threshold,
        cyclefactor=1.5,
        interactive=False,
        verbose=True,
        parallel=True,
        # startmodel parameters
        startmodel=startmodel,
        # automasking parameters
        usemask='auto-multithresh',  # use ALMA 12m(short) values
        noisethreshold=3.0,
        sidelobethreshold=1.0,
        lownoisethreshold=1.5,
        minbeamfrac=2.0,
        negativethreshold=1000.0,
        # restart parameters
        restart=restart,
        calcpsf=calcpsf,
        calcres=calcres,
    )
    workdir = '{0}.workdirectory'.format(imagename)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)


def test_clean_line_mosaic(spw, ext=None, startmodel=None):
    """
    Clean-line function for the G285 field, which overlaps at the half-power
    radius, so particular parameters are selected for this field different from
    the single-pointing targets.
    """
    targ = TARGETS['G285_mosaic']
    log_post(':: Running clean ({0}, {1})'.format(targ.name, spw.name))
    start_velo = calc_start_velo(targ, spw)
    imagename = 'images/{0}/{0}_{1}'.format(targ.name, spw.name)
    if ext is not None:
        imagename = '{0}_{1}'.format(imagename, ext)
    if startmodel:
        startmodel = 'gbt_cubes/for_joint/{0}_{1}.jypix'.format(targ.name, spw.name)
    # scale typical RMS of 1.8 mJy/beam per 0.158 km/s channel for NH3 (1,1) to
    # other SPWs
    rms = 1.8 * np.sqrt(0.158 / float(spw.velo_width.strip('km/s')))
    threshold = '{0:.4f}mJy'.format(rms)
    check_delete_image_files(imagename, parallel=True)
    tclean(
        vis=VIS_FILES,
        imagename=imagename,
        field=targ.field,
        spw=str(spw.spw_id),
        specmode='cube',
        outframe='lsrk',
        veltype='radio',
        restfreq=spw.restfreq,
        start=start_velo,
        nchan=spw.nchan,
        phasecenter='J2000 18h44m19.424 -4d02m00.503',
        imsize=[525, 375],  # 175 as, 350 "efficient size"
        cell='0.5arcsec',  # 3.4 as / 7
        # gridder parameters
        gridder='mosaic',
        # deconvolver parameters
        deconvolver='multiscale',
        scales=[0, 7, 21],  # point, 1, 3 beam hpbw's
        smallscalebias=0.6,
        restoringbeam='common',
        weighting='briggs',
        robust=1.0,
        niter=1000000,
        threshold=threshold,
        cyclefactor=1.5,
        interactive=False,
        verbose=True,
        parallel=True,
        # startmodel parameters
        startmodel=startmodel,
        # automasking parameters
        usemask='auto-multithresh',  # use ALMA 12m(short) values
        noisethreshold=3.0,
        sidelobethreshold=1.0,
        lownoisethreshold=1.5,
        minbeamfrac=2.0,
        negativethreshold=1000.0,
    )
    workdir = '{0}.workdirectory'.format(imagename)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)


def clean_all_lines(just_ammonia=False):
    if just_ammonia:
        spws = [SPWS['nh3_11'], SPWS['nh3_22']]
    else:
        spws = SPWS.values()
    for ii, targ in enumerate(TARGETS.values()):
        for jj, spw in enumerate(spws):
            log_post(':: {0} -- {1}'.format(targ.name, spw.name))
            log_post(':: Target {0:0>2d}'.format(ii+1))
            log_post(':: Line   {0:0>2d}'.format(jj+1))
            if targ.name == 'G285_mosaic':
                test_clean_line_mosaic(spw, ext='joint', startmodel=True)
            else:
                test_clean_line_target(targ, spw, ext='joint', startmodel=True)


###############################################################################
# Feather and single-dish combination
###############################################################################

def calc_offset_pix_shifts(targ):
    """
    Calculate the offset in radians from the catalog produced from
    `image_registration`. CASA typically (as far as I know) uses CDELT values
    in radians, but other software will produce FITS files using degrees.
    """
    filen = CAT_DIR + 'offsets_{0}_nh3_11.csv'.format(targ)
    arr = np.genfromtxt(filen, skip_header=1, delimiter=',')
    # file reports offsets in pixels, need to convert to angle,
    # further, CDELT header keyword will require value in rad
    arcsec_per_pix = 0.5
    rad_per_arcsec = 1.0 / 206264.806
    shift1 = np.nanmedian(arr[:,3]) * arcsec_per_pix * rad_per_arcsec
    shift2 = np.nanmedian(arr[:,4]) * arcsec_per_pix * rad_per_arcsec
    return shift1, shift2


def convert_img_k_to_jy(imagename, outfile):
    """
    Calculated from flux density / brightness temp conversion page:
      https://science.nrao.edu/facilities/vla/proposing/TBconv
    NOTE the implicit conversion at the top for (beam/omega) into [ster]

    Image must have units:
        restfreq -> Hz
        bmaj     -> arcsec
        bmin     -> arcsec
        cdelt1   -> rad

    Parameters
    ----------
    imagename : str
    outfile : str
    perbeam : bool, default True
        return in units of "Jy/beam", otherwise "Jy/beam" if False
    """
    # rest frequency for K to Jy conversion
    freq_d = imhead(imagename, mode='get', hdkey='restfreq')
    assert freq_d['unit'] == 'Hz'
    nu_ghz = freq_d['value'] / 1e9  # to GHz
    # beam major FWHM
    bmaj_d = imhead(imagename, mode='get', hdkey='bmaj')
    assert bmaj_d['unit'] == 'arcsec'
    thetamaj_as = bmaj_d['value']
    # beam minor FWHM
    bmin_d = imhead(imagename, mode='get', hdkey='bmin')
    assert bmin_d['unit'] == 'arcsec'
    thetamin_as = bmin_d['value']
    # pixel size, square pixels, from coordinate delta
    cdelt1 = imhead(imagename, mode='get', hdkey='cdelt1')
    assert cdelt1['unit'] == 'rad'
    pixsize_as = abs(cdelt1['value']) * 206264.8  # to arcsec
    # beam to pixel_size, prefactor = pi / (4 log(2))
    beamsize_as2 = 1.3309004 * thetamaj_as * thetamin_as
    pixperbeam = beamsize_as2 / pixsize_as**2
    # compute image in units of Jy/beam
    jybmimage = outfile + '.jybm'
    if_exists_remove(jybmimage)
    immath(
            imagename=imagename,
            expr='8.18249739e-7*{0:.6f}*{0:.6f}*IM0*{1:.6f}*{2:.6f}'.format(
                nu_ghz, thetamaj_as, thetamin_as),
            outfile=jybmimage,
    )
    imhead(jybmimage, mode='put', hdkey='bunit', hdvalue='Jy/beam')
    # compute image in units of Jy/pix
    jypiximage = outfile + '.jypix'
    if_exists_remove(jypiximage)
    immath(
            imagename=jybmimage,
            expr='IM0/{0:.6f}'.format(pixperbeam),
            outfile=jypiximage,
    )
    imhead(jypiximage, mode='put', hdkey='bunit', hdvalue='Jy/pixel')


def convert_sd_to_image():
    out_dir = GBT_DIR + 'for_joint/'
    #for targ in ('G23297',):
    #    for mol in ('nh3_11',):
    for targ in TARGETS.keys():
        if targ == 'G285_mosaic':
            continue
        for mol in ('nh3_11', 'nh3_22'):
            stem = '{0}_{1}'.format(targ, mol)
            ## fitsimage -> casaimage
            # convert to a CASA image
            fitsimage = GBT_DIR + '{0}/{0}_{1}_vavg.fits'.format(targ, mol.upper())
            casaimage = out_dir + '{0}_{1}.image'.format(targ, mol)
            importfits(fitsimage=fitsimage, imagename=casaimage, overwrite=True)
            ## casaimage -> etambimage
            # put the GBT data on the main-beam scale with etamb=0.75
            etambimage = out_dir + stem + '.etamb'
            if_exists_remove(etambimage)
            immath(
                    imagename=casaimage,
                    expr='IM0/0.75',
                    outfile=etambimage,
            )
            # shift reference pixel to correct for pointing errors
            shift1, shift2 = calc_offset_pix_shifts(targ)
            cdelt1 = imhead(etambimage, mode='get', hdkey='CDELT1')
            cdelt2 = imhead(etambimage, mode='get', hdkey='CDELT2')
            assert cdelt1['unit'] == 'rad' and cdelt2['unit'] == 'rad'
            crpix1 = imhead(etambimage, mode='get', hdkey='CRPIX1')
            crpix2 = imhead(etambimage, mode='get', hdkey='CRPIX2')
            new_crpix1 = crpix1 - shift1 / cdelt1['value']
            new_crpix2 = crpix2 - shift2 / cdelt2['value']
            imhead(etambimage, mode='put', hdkey='CRPIX1', hdvalue=new_crpix1)
            imhead(etambimage, mode='put', hdkey='CRPIX2', hdvalue=new_crpix2)
            ## etambimage -> regridimage
            # regrid the SD image to the VLA image spatial resolution
            vlaimage = IMG_DIR + '{0}/{0}_{1}.image'.format(targ, mol)
            regridimage = out_dir + stem + '.regrid'
            imregrid(
                    imagename=etambimage,
                    template=vlaimage,
                    axes=[0, 1],
                    interpolation='cubic',
                    output=regridimage,
                    overwrite=True,
            )
            ## regridimage -> stokesimage
            # the spectral averaged cube lost the stokes param, so put back
            stokesimage = out_dir + stem + '.withstokes'
            if_exists_remove(stokesimage)
            ia.open(regridimage)
            im_r = ia.adddegaxes(outfile=stokesimage, stokes='I')
            im_r.done()
            ia.close()
            ## stokesimage -> reordimage
            # reorder the axes because of difference in velocity/stokes
            #   VLA images have -> ra/dec/stokes/velo
            #   GBT images have -> ra/dec/velo/stokes
            reordimage = out_dir + stem + '.reord'
            if_exists_remove(reordimage)
            imtrans(imagename=stokesimage, outfile=reordimage, order='0132')
            ## reordimage -> depbimage
            # multiply for primary beam attenuation of the VLA image
            vlapb = IMG_DIR + '{0}/{0}_{1}.pb'.format(targ, mol)
            depbimage = out_dir + stem + '.depb'
            if_exists_remove(depbimage)
            immath(
                    imagename=[reordimage, vlapb],
                    expr='IM0*IM1',
                    outfile=depbimage,
                    imagemd=reordimage,  # to get meta-data right
            )
            ## depbimage -> jybimage, jypiximage
            # convert to units of Jy/pixel, which is needed for startmodel
            jyimage_base = out_dir + stem
            convert_img_k_to_jy(depbimage, jyimage_base)


def process_image_for_feather(targ, spw):
    names = (targ.name, spw.name)
    log_post(':: Processing {0}_{1} for feather'.format(*names))
    imagebase = 'images/{0}/{0}_{1}_joint.image'.format(*names)
    concatimage = imagebase + '.concat'
    commonimage = imagebase + '.common'
    # image produced by parallel clean must be concat'd into one
    # to work properly with feather and the GBT data
    concat_parallel_image(imagebase)
    # image has per-channel beams, convert to single common beam
    if_exists_remove(commonimage)
    imsmooth(concatimage, kernel='common', outfile=commonimage)


def joint_post_feather_image(targ, spw):
    names = (targ.name, spw.name)
    log_post(':: Feathering {0}_{1}'.format(*names))
    vlaimage = 'images/{0}/{0}_{1}_joint.image.common'.format(*names)
    gbtimage = 'gbt_cubes/for_joint/{0}_{1}.jybm'.format(*names)
    fthimage = 'images/{0}/{0}_{1}_jfeather.image'.format(*names)
    if_exists_remove(fthimage)
    feather(fthimage, lowres=gbtimage, highres=vlaimage)


def joint_post_feather_all():
    for targ in TARGETS.values():
        for spw in (SPWS['nh3_11'], SPWS['nh3_22']):
            process_image_for_feather(targ, spw)
            joint_post_feather_image(targ, spw)


###############################################################################
# Imaging tests
###############################################################################

def test_startmodel():
    """
    Test the startmodel approach for jointly imaging the single-dish data.
    """
    targ = TARGETS['G28539']
    spw = SPWS['nh3_11']
    log_post(':: Running clean ({0}, {1})'.format(targ.name, spw.name))
    start_velo = calc_start_velo(targ, spw)
    imagename = 'images/{0}/{0}_{1}'.format(targ.name, spw.name)
    # scale typical RMS of 1.8 mJy/beam per 0.158 km/s channel for NH3 (1,1) to
    # other SPWs
    rms = 1.8 * np.sqrt(0.158 / float(spw.velo_width.strip('km/s')))
    threshold = '{0:.4f}mJy'.format(rms)
    imagename = 'test_imaging/test_startmodel/G28539_joint'
    check_delete_image_files(imagename)
    tclean(
        vis=VIS_FILES,
        imagename=imagename,
        field=targ.field,
        spw=str(spw.spw_id),
        specmode='cube',
        outframe='lsrk',
        veltype='radio',
        restfreq=spw.restfreq,
        start=start_velo,
        nchan=spw.nchan,
        imsize=[350, 350],  # 175 as, 350 "efficient size"
        cell='0.5arcsec',  # 3.4 as / 7
        # gridder parameters
        gridder='standard',
        # deconvolver parameters
        deconvolver='multiscale',
        scales=[0, 7, 21],  # point, 1, 3 beam hpbw's
        smallscalebias=0.6,
        restoringbeam='common',
        weighting='briggs',
        robust=1.0,
        niter=1000000,
        threshold=threshold,
        cyclefactor=1.5,
        interactive=False,
        verbose=True,
        parallel=True,
        # startmodel
        startmodel='test_imaging/test_startmodel/G28539_nh3_11.jypix',
        # automasking parameters
        usemask='auto-multithresh',  # use ALMA 12m(short) values
        noisethreshold=3.0,
        sidelobethreshold=1.0,
        lownoisethreshold=1.5,
        minbeamfrac=2.0,
        negativethreshold=1000.0,
    )
    workdir = '{0}.workdirectory'.format(imagename)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)


###############################################################################
# Moment maps
###############################################################################

def make_ia_moments(targ, spw):
    outfile_fmt = 'moments/{name}/{name}_{line}_snr{snr}_smooth{smooth}'
    maxv = 200  # sigma, need upper bound for range argument
    smoothaxes = [0, 1, 3]  # ra, dec, velo
    smoothtypes = ['gauss', 'gauss', 'hann']
    # FWHM of kernel in 0.5 pixels -> 10.5 arcsec
    # 3 pixels for hanning smooth of spectral
    smoothwidths = [21, 21, 3]
    velowidth = 3  # km/s, radius of window
    xypix = (525, 375) if targ.name.endswith('mosaic') else (350, 350)
    region = (
            'box[[0pix,0pix],[{0}pix,{1}pix]], '.format(*xypix) +
            'range=[{0:.2f}km/s,{1:.2f}km/s]'.format(
                    targ.vlsr - velowidth, targ.vlsr + velowidth)
    )
    imagename_base = 'images/{0}/{0}_{1}'.format(targ.name, spw.name)
    if spw.name in ('nh3_11', 'nh3_22'):
        imagename = imagename_base + '_jfeather.image'
    else:
        imagename = imagename_base + '.image'
    # scale typical RMS of 1.8 mJy/beam per 0.158 km/s channel for NH3
    # (1,1) to other SPWs. mJy/beam -> 1e-3 Jy/bm
    rms = 1.8e-3 * np.sqrt(0.158 / float(spw.velo_width.strip('km/s')))
    ia.open(imagename)
    for snr in (-10, 1, 2, 3, 4):
        # unsmoothed moments
        ia.moments(
            moments=[-1,0,1,2,8],
            region=region,
            axis=3,
            includepix=[snr*rms,maxv],
            outfile=outfile_fmt.format(
                    name=targ.name, line=spw.name, snr=str(snr),
                    smooth='0'),
            overwrite=True,
        ).done()
        # smoothed moments
        # the RMS will change from smoothing, so modify threshold
        # ratio of beam sizes in pixels times a sqrt(2) factor from
        # Hanning smooth in velocity
        eta = smoothwidths[0] / 7 * np.sqrt(2)
        ia.moments(
            moments=[-1,0,1,2,8],
            region=region,
            axis=3,
            includepix=[snr*rms/eta,maxv],
            smoothaxes=smoothaxes,
            smoothtypes=smoothtypes,
            smoothwidths=smoothwidths,
            outfile=outfile_fmt.format(
                name=targ.name, line=spw.name, snr=str(snr),
                smooth=str(smoothwidths[0])),
            overwrite=True,
        ).done()
    ia.close()


def make_all_moments():
    for targ in TARGETS.values():
        for spw in SPWS.values():
            if targ.name == 'G285_mosaic' and spw.name not in ('nh3_11', 'nh3_22'):
                continue
            log_post('-- Moments {0}_{1}'.format(targ.name, spw.name))
            make_ia_moments(targ, spw)


def pbcor_moments(targ, overwrite=True):
    log_post(':: Primary beam correcting moment maps')
    spws = (SPWS['nh3_11'], SPWS['nh3_22'])  # XXX
    #for spw in SPWS.values():
    for spw in spws:  # XXX
        nh3_lines = ('nh3_11', 'nh3_22')
        if targ.name == 'G285_mosaic' and spw.name not in nh3_lines:
            continue
        log_post('-- Correcting {0}_{1}'.format(targ.name, spw.name))
        line = spw.name+'_joint' if spw.name in nh3_lines else spw.name
        pbimage = 'images/{0}/{0}_{1}.pb'.format(targ.name, line)
        # NOTE We need to remove spectral axis in the primary beam image in
        # order for both data to have the same shape.  Luckily, `impbcor` can
        # be fed an array as well as a file name.
        ia.open(pbimage)
        pbdata = ia.getregion()[...,0]  # beam of channel number 0
        ia.close()
        infiles = glob('moments/{0}/{0}_{1}_*'.format(targ.name, spw.name))
        infiles = [
                s for s in infiles
                if s.endswith('average')
                or s.endswith('integrated')
                or s.endswith('maximum')
        ]
        for imagename in infiles:
            impbcor(imagename=imagename, pbimage=pbdata,
                    outfile=imagename+'.pbcor', overwrite=overwrite)


def pbcor_all_moments():
    for targ in TARGETS.values():
        pbcor_moments(targ)


def export_moments(targ):
    infiles = [
        s for s in glob('moments/{0}/{0}_*'.format(targ.name))
        if not s.endswith('.fits')
    ]
    for imagename in infiles:
        export_fits(imagename)


def export_all_moments():
    for targ in TARGETS.values():
        export_moments(targ)


###############################################################################
# Gaincal tests
###############################################################################

VIS_FILES_TSELFCAL = sorted(glob(
    ROOT_DIR+'test_imaging/test_selfcal/test_split_eb*.ms'
))


def test_populate_model_column():
    """
    Fill the model column using a start-model and an niter=0 with the savemodel
    parameter. Important params:
        (startmodel='foo.model', niter=0, savemodel='modelcolumn')
    """
    targ = TARGETS['G28539']
    spw = SPWS['nh3_11']
    test_path = 'test_imaging/test_selfcal/'
    vis = test_path + 'test_split_1eb.ms'
    startmodel = test_path + 'full_data_nh3_11.model.concat'
    imagename = test_path + 'G28539_1eb'
    start_velo = calc_start_velo(targ, spw)
    for vis in VIS_FILES_TSELFCAL:
        clearcal(vis=vis, addmodel=True)
    check_delete_image_files(imagename)
    tclean(
        vis=VIS_FILES_TSELFCAL,
        imagename=imagename,
        field='0',
        spw='0',
        specmode='cube',
        outframe='lsrk',
        veltype='radio',
        restfreq=spw.restfreq,
        start=start_velo,
        nchan=spw.nchan,
        imsize=[350, 350],  # 175 as, 350 "efficient size"
        cell='0.5arcsec',  # 3.4 as / 7
        # gridder parameters
        gridder='standard',
        # deconvolver parameters
        niter=0,
        threshold='5mJy',
        interactive=False,
        verbose=True,
        # startmodel parameters
        startmodel=startmodel,
        savemodel='modelcolumn',
    )


def test_gaincal_pcal1():
    test_path = 'test_imaging/test_selfcal/'
    for vis in VIS_FILES_TSELFCAL:
        caltable = vis + '.pcal1_30s.cal'
        cals = []
        gaincal(
                vis=vis,
                caltable=caltable,
                gaintable=cals,
                gaintype='G',
                solint='30s',
                combine='',
                #solnorm=True,
                calmode='p',
        )


def test_applycal_pcal1():
    for vis in VIS_FILES_TSELFCAL:
        gaintable = [vis+'.pcal1.cal']
        clearcal(vis=vis, addmodel=True)
        applycal(
                vis=vis,
                gainfield='0',
                gaintable=gaintable,
                interp='linear',
                applymode='calonly',
                calwt=False,
        )


def test_image_pcal1_results():
    """
    Fill the model column using a start-model and an niter=0 with the savemodel
    parameter. Important params:
        (startmodel='foo.model', niter=0, savemodel='modelcolumn')
    """
    targ = TARGETS['G28539']
    spw = SPWS['nh3_11']
    test_path = 'test_imaging/test_selfcal/'
    imagename = test_path + 'G28539_pcal1_30s'
    start_velo = calc_start_velo(targ, spw)
    check_delete_image_files(imagename, parallel=True)
    tclean(
        vis=VIS_FILES_TSELFCAL,
        imagename=imagename,
        field='0',
        spw='0',
        specmode='cube',
        outframe='lsrk',
        veltype='radio',
        restfreq=spw.restfreq,
        start=start_velo,
        nchan=spw.nchan,
        imsize=[350, 350],  # 175 as, 350 "efficient size"
        cell='0.5arcsec',  # 3.4 as / 7
        # gridder parameters
        gridder='standard',
        # deconvolver parameters
        deconvolver='multiscale',
        scales=[0, 7, 21],  # point, 1, 3 beam hpbw's
        smallscalebias=0.6,
        restoringbeam='common',
        weighting='briggs',
        robust=1.0,
        niter=1000000,
        threshold='1.8mJy',
        cyclefactor=1.5,
        interactive=False,
        verbose=True,
        parallel=True,
        # automasking parameters
        usemask='auto-multithresh',  # use ALMA 12m(short) values
        noisethreshold=3.0,
        sidelobethreshold=1.0,
        lownoisethreshold=1.5,
        minbeamfrac=2.0,
        negativethreshold=1000.0,
        # startmodel parameters
        #savemodel='modelcolumn',
        datacolumn='corrected',
    )


