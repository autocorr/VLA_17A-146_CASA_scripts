"""
=================
Iterative Feather
=================

An iterative method to combine single-dish data with interferometric data in the Fourier domain using feather/immerge. The values are tuned for VLA 17A-146, which is narrowband, spectral-line imaging of single-pointing fields.
"""

from __future__ import (print_function, division)

import os
import shutil
import datetime
from glob import glob
from copy import deepcopy
from collections import namedtuple

import numpy as np

from imagerhelpers.imager_base import PySynthesisImager
from imagerhelpers.imager_parallel_cube import PyParallelCubeSynthesisImager
from imagerhelpers.input_parameters import ImagerParameters


############################################################################
# Global parameters specific to 17A-146
############################################################################

ia.close()
ia.done()
ib = iatool()

FWHM = 2 * np.sqrt(2 * np.log(2))

ROOT_DIR = '/lustre/aoc/users/bsvoboda/17A-146/data/'
VIS_FILES = sorted(glob('17A-146_*/pipe_specline/17A-146.sb*.ms'))
VIS_FILES = [ROOT_DIR + s for s in VIS_FILES]

TEST_DIR = 'test_imaging/test_iterfeather/'
TEST_IMG = TEST_DIR + 'G28539_nh3_11_iterf'
SD_IMG   = TEST_DIR + 'G28539_nh3_11.jybm'

PARAMS = ImagerParameters(
    msname=VIS_FILES,
    imagename=TEST_IMG,
    field='7',  # G28539
    spw='0',  # NH3 (1,1)
    specmode='cube',
    outframe='lsrk',
    veltype='radio',
    restfreq='23.6944955GHz',
    start='58.58km/s',
    nchan=380,
    imsize=[350, 350],  # 175 as, 350 "efficient size"
    cell='0.5arcsec',  # 3.4 as / 7
    pblimit=0.2,
    # gridder parameters
    gridder='standard',
    # deconvolver parameters
    deconvolver='multiscale',
    scales=[0, 7, 21],  # point, 1, 3 beam hpbw's
    scalebias=0.6,
    restoringbeam='common',
    weighting='briggs',
    robust=1.0,
    niter=1000000,
    threshold='1.8mJy',
    cyclefactor=1.5,
    interactive=False,
    verbose=True,
    # automasking parameters
    usemask='auto-multithresh',  # use ALMA 12m(short) values
    noisethreshold=3.0,
    sidelobethreshold=1.0,
    lownoisethreshold=1.5,
    minbeamfrac=2.0,
    negativethreshold=1000.0,
)


###############################################################################
# General utility functions
###############################################################################

def log_post(msg):
    """
    Post a message to the CASA logger and logfile.
    """
    casalog.post(msg, 'INFO', 'bsvoboda')


def if_exists_remove(imagename):
    if os.path.exists(imagename):
        log_post(':: Removing {0}'.format(imagename))
        try:
            rmtables(imagename)
            shutil.rmtree(imagename)
        except OSError:
            pass


def hard_remove(imagename):
    try:
        shutil.rmtree(imagename)
        log_post(':: Removed {0}'.format(imagename))
    except OSError:
        pass


def remove_all_extensions(imagename):
    for filen in glob('{0}.*'.format(imagename)):
        if_exists_remove(filen)


def convert_to_common_beam(imagename):
    log_post(':: Converting to common beam for {0}'.format(imagename))
    intermediate = imagename + '.intermediate'
    if_exists_remove(intermediate)
    imsmooth(imagename, kernel='common', outfile=intermediate)
    rmtables(imagename)
    shutil.move(intermediate, imagename)
    # get restoring beam from new PSF file
    ia.open(imagename)
    beam = ia.commonbeam()
    beam['positionangle'] = beam['pa']
    del beam['pa']
    ia.close()
    return beam


def get_data_from_image(imagename):
    ia.open(imagename)
    data = ia.getchunk()
    ia.close()
    return data


def write_data_into_image(imagename, data):
    ia.open(imagename)
    imageshape = ia.shape()
    imagedata = ia.getchunk()
    assert np.all(imageshape == np.array(data.shape))
    beam = ia.restoringbeam()
    ia.setrestoringbeam(remove=True)
    ia.putchunk(data)
    ia.setrestoringbeam(beam=beam)
    ia.close()


###############################################################################
# Image reconstruction with PySynthesisImager
###############################################################################

def make_psf_from_cube(sdimage):
    header = imhead(sdimage, mode='list')
    shape = header['shape']
    # calculate beam standard deviation in pixels
    ia.open(sdimage)
    beam = ia.restoringbeam()
    ia.close()
    bmaj = beam['major']['value']
    cdelt1 = header['cdelt1']  # default value is in radian
    cell = qa.abs(qa.convert(cdelt1, 'arcsec'))['value']
    sigpix = bmaj / cell / FWHM
    # write PSF data directly to data-field of image
    log_post(':: Writing PSF data for {0}'.format(imagename))
    psfname = '{0}.psf'.format(sdimage)
    if_exists_remove(psfname)
    shutil.copytree(sdimage, psfname)
    ia.open(psfname)
    imgdata = ia.getchunk()
    ax0_size, ax1_size = imgdata.shape[:2]
    ax0, ax1 = np.meshgrid(np.arange(ax0_size), np.arange(ax1_size))
    psf = np.exp(-(
            (ax0 - ax0_size/2.0)**2 / (2 * sigpix**2)
          + (ax1 - ax1_size/2.0)**2 / (2 * sigpix**2)
    ))
    psf = psf[..., np.newaxis, np.newaxis] * np.ones(imgdata.shape)
    ia.putchunk(psf)
    imhead(psfname, mode='put', hdkey='bunit', hdvalue='')
    ia.close()
    # formatting per plane beams
    #log_post(':: Formatting per plane beams for {0}'.format(imagename))
    #assert header['ctype4'] == 'Frequency'
    #nchan = shape[3]
    #ia.setrestoringbeam(remove=True)
    #for ii in range(nchan):
    #    ia.setrestoringbeam(beam=beam, channel=ii)
    #ia.close()


def init_cube_imager(params):
    """
    Initialize the Imager module and create a dirty map to use as the first
    residual.

    Parameters
    ----------
    params : imagerhelpers.input_parameters.ImagerParameters

    Returns
    -------
    imager : imagerhelpers.imager_base.PySynthesisImager
    """
    log_post(':: Initializing cube imager')
    # FIXME replace with PyParallelCubeSynthesisImager
    imager = PySynthesisImager(params=params)
    imagename = imager.allimpars['0']['imagename']
    remove_all_extensions(imagename)
    # Initialize the major cycle modules
    imager.initializeImagers()
    imager.initializeNormalizers()
    imager.setWeighting()
    imager.initializeDeconvolvers()
    imager.initializeIterationControl()
    # Initialize the minor cycle modules
    imager.makePSF()
    imager.makePB()
    # Create initial dirty image and residual image
    imager.runMajorCycle()
    return imager


def do_reconstruct(params, sdimage=SD_IMG, method='iterf'):
    """
    Do joint image reconstruction with single-dish and interferometric data.
    This function uses the PySynthesisImager refactored tclean API and takes a
    ImagerParameters instance. Two methods are implemented for methods of
    "combine during deconvolution" or CBD: (1) an interative image plane
    addition method adapted from Stanimirovic et al. (1999) and (2) an
    iterative feather approach described in Rao et al. (in prep).

    Parameters
    ----------
    params : ImagerParameters
    sdimage : str
    method : str, default='iterf'
        Image combination method:
            'stanim' -> image plane addition (Stanimirovic et al. 1999)
            'iterf'  -> iterative feather (Rao et al. in prep)
    """
    combine_map = {'iterf': feather_sd, 'stanim': add_sd}
    combine_sd = combine_map[method]
    imager = init_cube_imager(params)
    imagename = imager.allimpars['0']['imagename']
    # Feather the PSF and dirty map with the SD image
    init_sd_residual(sdimage)
    beam = convert_to_common_beam(imagename+'.psf')
    combine_sd(imagename, sdimage, beam, ext='psf')
    combine_sd(imagename, sdimage, beam, ext='residual')
    # Create first clean mask
    imager.hasConverged()
    imager.updateMask()
    while not imager.hasConverged():
        imager.runMinorCycle()
        imager.runMajorCycle()
        calc_sd_residual(imagename, sdimage)
        combine_sd(imagename, sdimage, beam, ext='residual')
        imager.updateMask()
    imager.restoreImages()
    imager.pbcorImages()
    imager.deleteTools()


def init_sd_residual(sdimage):
    log_post(':: Initializing single dish residual image')
    residual = sdimage + '.residual'
    if_exists_remove(residual)
    shutil.copytree(sdimage, residual)


def fix_int_restoring_beam(imagename, beam, ext='residual'):
    filename = '{0}.{1}'.format(imagename, ext)
    log_post(':: Fixing restoring beam in residual {0}'.format(filename))
    ia.open(filename)
    ia.setrestoringbeam(remove=True)
    ia.setrestoringbeam(beam=beam)
    ia.close()


def feather_sd(imagename, sdimage, beam, ext='residual', sdfactor=1.0):
    log_post(':: Feather single-dish residuals')
    highres = '{0}.{1}'.format(imagename, ext)
    lowres = '{0}.{1}'.format(sdimage, ext)
    fix_int_restoring_beam(imagename, beam, ext=ext)
    highres_interm = highres + '.intermediate'
    if_exists_remove(highres_interm)
    feather(
            imagename=highres_interm,
            highres=highres,
            lowres=lowres,
            sdfactor=sdfactor,
    )
    data = get_data_from_image(highres_interm)
    write_data_into_image(highres, data)


def add_sd(imagename, sdimage, beam, ext='residual', sdfactor=1.0):
    log_post(':: Add single-dish residual image')
    highres = '{0}.{1}'.format(imagename, ext)
    lowres = '{0}.{1}'.format(sdimage, ext)
    fix_int_restoring_beam(imagename, beam, ext=ext)
    highres_interm = highres + '.intermediate'
    alpha = calculate_alpha(highres, lowres)
    fscale = sdfactor if ext == 'residual' else 1.0
    expr = '(IM0+{0:.7f}*{1:.7f}*IM1)/(1+{1:.7f})'.format(fscale, alpha)
    if_exists_remove(highres_interm)
    immath(
            imagename=[highres, lowres],
            outfile=highres_interm,
            mode='evalexpr',
            expr=expr,
    )
    data = get_data_from_image(highres_interm)
    write_data_into_image(highres, data)


def calculate_alpha(highres, lowres):
    ia.open(highres)
    in_bm = ia.restoringbeam()
    ia.close()
    ia.open(lowres)
    sd_bm = ia.restoringbeam()
    ia.close()
    in_omega = in_bm['major']['value'] * in_bm['minor']['value']
    sd_omega = sd_bm['major']['value'] * sd_bm['minor']['value']
    alpha = in_omega / sd_omega
    return alpha


def calc_sd_residual(imagename, sdimage, sdgain=1.0):
    modelname = '{0}.model'.format(imagename)
    convmodelname = '{0}.model.convolved'.format(imagename)
    bmaj = imhead(sdimage, mode='get', hdkey='bmaj')['value']
    bmin = imhead(sdimage, mode='get', hdkey='bmin')['value']
    log_post(':: Convolve model')
    if_exists_remove(convmodelname)
    # convolve the model to single-dish angular resolution
    imsmooth(
            imagename=modelname,
            outfile=convmodelname,
            kernel='gauss',
            major='{0:.2f}arcsec'.format(bmaj),
            minor='{0:.2f}arcsec'.format(bmin),
            pa='0deg',
    )
    # substract the smoothed model from the observed single-dish image to make
    # the single-dish residual
    log_post(':: Subtracting model from single-dish')
    residualname = '{0}.residual'.format(sdimage)
    if_exists_remove(residualname)
    immath(
            imagename=[sdimage, convmodelname],
            outfile=residualname,
            mode='evalexpr',
            expr='IM0-IM1',
    )


