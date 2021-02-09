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
import glob
import shutil
import datetime
from collections import namedtuple
import numpy as np
#import findContinuum  # probably only on NAASC machines


Spw = namedtuple('Spw', 'name, spw_id, restfreq, velo_width, nchan, ot_name')
SPWS = { spw.name : spw for spw in [
    Spw('cs',    1,  '97980.953MHz', '48.0km/s',     6, '#BB_3#SW-01'),
    Spw('so',    1,  '99299.870MHz', '48.0km/s',     6, '#BB_3#SW-01'),
    Spw('hc3n',  2, '100076.391MHz', '48.0km/s',     6, '#BB_4#SW-01'),
    Spw('hcop',  3,  '89188.526MHz',  '0.106km/s', 200, '#BB_1#SW-01'),
    Spw('hcn',   4,  '88631.847MHz',  '0.106km/s', 300, '#BB_1#SW-02'),
    Spw('nhhd',  5,  '85926.258MHz',  '0.107km/s', 150, '#BB_2#SW-01'),
    Spw('htcop', 6,  '86754.288MHz',  '0.106km/s', 100, '#BB_2#SW-02'),
]}
# NOTE: continuum TDM's: '#BB_3#SW-01' and '#BB_4#SW-01'


VELOS = {
    # name   vlsr [km/s]
    'G2432': 31.07,
    'G2984': 18.42,
    'G3302': 66.37,
    'G3604': 51.51,
    'G4029': 81.51,
}


LINE_CHANS_FMT = {
    'G2432': '{0}:16~23:101~107,{1}:23~30',
    'G2984': '{0}:16~23:101~107,{1}:23~30',
    'G3302': '{0}:16~23:101~107,{1}:23~30',
    'G3604': '{0}:16~23:101~107,{1}:23~30',
    'G4029': '{0}:16~23:101~107,{1}:23~30',
}


class ImagingConfig(object):
    line_chans_fmt = LINE_CHANS_FMT
    path_base_fmt = '{0}/calibrated_{0}'
    imsize = [500, 500]
    cell = '0.23arcsec'
    gridder = 'standard'
    deconvolver = 'hogbom'
    scales = None
    robust = 1.0
    refant = 'DA49'
    n_sb = 2
    n_spw = 6
    cont_spw_id = 0
    spw_id_start = 0

    def __init__(self, name):
        assert name in VELOS
        self.name = name
        self.path_base = self.path_base_fmt.format(name)
        self.vis = self.path_base + '.ms'
        self.viscs = self.vis + '.contsub'
        self.vislf = self.path_base + '_linefree.ms'
        self.vlsr = VELOS[name]

    @property
    def line_chans(self):
        line_chans_fmt = self.line_chans_fmt[self.name]
        spw_ids = np.arange(self.spw_id_start, self.n_spw)
        line_chans = ','.join([
            # "splat" the array into the positional args of the format string
            line_chans_fmt.format(*(self.n_spw * ii + spw_ids))
            for ii in range(self.n_sb)
        ])
        return line_chans

    @property
    def chan_widths(self):
        width = self.n_sb * [
            8,    # 0, continuum  98 GHz
            8,    # 1, continuum 100 GHz
        ]
        # ^ NOTE that these are the number of native channels wide the new
        #        channels will be. Ex., the 128 continuum channels will be
        #        averaged into 16 channels that are 8 native channels wide.
        return width

    def start_vlsr(self, spw):
        half_width = float(spw.velo_width.strip('km/s')) * spw.nchan / 2
        start_velo = VELOS[self.name] - half_width
        return '{0:.2f}km/s'.format(start_velo)

    def get_one_spw(self, spw, contsub=False):
        vis = self.viscs if contsub else self.vis
        return get_spw_from_name(vis, spw)

    def get_cont_spw(self, linefree=False):
        vis = self.vislf if linefree else self.vis
        return get_spw_from_name(vis, spw=None)



class ImagingConfigAca(ImagingConfig):
    line_chans_fmt = LINE_CHANS_FMT
    path_base_fmt = '{0}/calibrated_{0}_7m'
    imsize = [144, 144]
    cell = '1.3arcsec'
    refant = 'CM03'
    n_sb = 9
    cont_spw_id = 4
    spw_id_start = 0

    @property
    def chan_widths(self):
        width = super(ImagingConfigAca, self).chan_widths
        return width[2:] + width[:2]


class ImagingConfigJoint(ImagingConfig):
    line_chans_fmt = LINE_CHANS_FMT
    path_base_fmt = '{0}/calibrated_{0}_joint'
    imsize = [250, 250]
    cell = '0.46arcsec'
    gridder = 'mosaic'
    deconvolver = 'multiscale'
    scales = [0, 7, 14, 28]
    smallscalebias = 0.6
    robust = 1.0
    refant = ['DA49', 'CM03']
    n_sb = 11
    cont_spw_id = None
    spw_id_start = None


def get_joint_pair(name):
    return [ImagingConfig(name), ImagingConfigAca(name)]


def get_spw_from_name(vis, spw=None):
    """
    Use the `msmd` tool to identify the individual SPW ID numbers for a common
    baseband and spectral window pair, which should always be consistent between
    the 12m & 7m MSs, as setup by the OT.

    Parameters
    ----------
    vis : str
    spw : namedtuple (from `SPWS`), default None
        An SPW object, or `None` for the continuum

    Returns
    -------
    spws : str
        Comma seperated list of the ID numbers for each SPW
    """
    msmd.open(vis)
    labels = msmd.namesforspws()
    if spw is None:
        ot_name = '#BB_4#SW-01'
    if type(spw) is str:
        ot_name = spw
    else:
        ot_name = spw.ot_name
    ot_name = ot_name + '#FULL_RES'
    spws = ','.join([
        str(i) for i,s in enumerate(labels) if ot_name in s
    ])
    msmd.close()
    return spws


###############################################################################
# General utility functions
###############################################################################

def check_delete_image_files(imagename, preserve_mask=False):
    """
    Check for and remove (if they exist) files created by clean such as '.flux',
    '.image', etc.

    Parameters
    ----------
    imagename : str
        The relative path name to the files to delete.
    preserve_mask : bool, default False
        Whether to preserve the `.mask` file extension
    """
    print(':: Check for and remove existing files')
    exts = [
        '.flux', '.pb', '.image', '.weight', '.model', '.pbcor', '.psf',
        '.sumwt', '.residual', '.flux.pbcoverage'
    ]
    if not preserve_mask:
        exts += ['.mask']
    for ext in exts:
        if os.path.exists(imagename+ext):
            filen = imagename + ext
            print('-- Removing {0}'.format(filen))
            rmtables(filen)
    if os.path.exists(imagename+'.residual'):
        print('-- Hard remove .residual')
        shutil.rmtree(imagename+'.residual')


def export_fits(imagename):
    print(':: Exporting fits')
    exportfits(imagename, imagename+'.fits', velocity=True, overwrite=True)


def concat_arrays(imcl, linefree=False, contsub=False, overwrite=True):
    print(':: Check for and remove existing files')
    if linefree:
        concatvis = imcl[0].path_base+'_joint_linefree.ms'
    elif contsub:
        concatvis = imcl[0].path_base+'_joint.ms.contsub'
    else:
        concatvis = imcl[0].path_base+'_joint.ms'
    if overwrite & os.path.exists(concatvis):
        print('-- Removing {0}'.format(concatvis))
        rmtables(concatvis)
    print(':: Concatenating ms files')
    if linefree:
        concat(vis=[imcl[0].vislf, imcl[1].vislf], concatvis=concatvis)
    elif contsub:
        concat(vis=[imcl[0].viscs, imcl[1].viscs], concatvis=concatvis)
    else:
        concat(vis=[imcl[0].vis, imcl[1].vis], concatvis=concatvis)


def primary_beam_correct(imagebase, overwrite=True, export=True):
    print(':: Check for and remove existing files')
    imagename  = imagebase + '.image'
    pbimage    = imagebase + '.pb'
    pbcorimage = imagebase + '.pbcor'
    impbcor(imagename=imagename, pbimage=pbimage, outfile=pbcorimage,
            overwrite=overwrite)
    if export:
        export_fits(pbcorimage)


###############################################################################
# Continuum imaging
###############################################################################

def clean_continuum(imc, linefree=False, remove_existing=True, export=False):
    """
    Image the continuum using only the single continuum TDM spectral window, or
    the linefree channels.

    Parameters
    ----------
    imc : ImagingConfig
    linefree : bool, default False
    """
    if linefree:
        vis = imc.vislf
        spw = ''
        imagename = imc.path_base
    else:
        vis = imc.vis
        spw = imc.get_cont_spw()
        imagename = imc.path_base +'_single'
    if remove_existing:
        check_delete_image_files(imagename)
    print(':: Running clean')
    tclean(vis=vis,
           imagename=imagename,
           spw=spw,
           specmode='mfs',
           imsize=imc.imsize,
           cell=imc.cell,
           gridder=imc.gridder,
           deconvolver=imc.deconvolver,
           restoringbeam='common',
           weighting='briggs',
           robust=imc.robust,
           niter=100000,
           interactive=True)
    if export:
        export_fits(imagename+'.image')


def split_linefree(imc):
    """
    Split out the line-free continuum channels from the measurement set and
    apply channel averaging (for the higher resolution line channels). The
    flags are backed up, line channels flagged, measurement set split, and the
    flag state restored. It should be OK to initialize the weight each time,
    since when found will no-op.

    Parameters
    ----------
    source_name : ImagingConfig
    """
    now = str(datetime.datetime.utcnow())
    comment = 'Flags before split for line-free (UTC: {0})'.format(now)
    print(':: Saving flag backup : `before_split`')
    flagmanager(vis=imc.vis, mode='save', versionname='before_split',
                comment=comment)
    initweights(vis=imc.vis, wtmode='weight', dowtsp=True)
    print(':: Flagging')
    flagdata(vis=imc.vis, mode='manual', spw=imc.line_chans)
    print(':: Split out and channel average')
    if os.path.exists(imc.vislf):
        print(':: Deleting existing split ms')
        try:
            shutil.rmtree(imc.vislf+'.flagversions')
        except OSError:
            pass
        rmtables(imc.vislf)
    split(vis=imc.vis, outputvis=imc.vislf, width=imc.chan_widths, datacolumn='data')
    print(':: Restoring flag backup')
    flagmanager(vis=imc.vis, mode='restore', versionname='before_split')
    flagmanager(vis=imc.vis, mode='delete', versionname='before_split')


###############################################################################
# Self calibration
###############################################################################

def selfcal_image(imc, trial='0'):
    """
    Self Calibration. Initiate the self-calibration process with a shallow clean
    to start. Here clean only the secure detections of emission.
    """
    if trial == '0':
        mask = ''
    else:
        mask = imc.path_base + '_p{0}.mask'.format(int(trial)-1)
    vis = imc.vislf
    imagename = imc.path_base + '_p{0}'.format(trial)
    check_delete_image_files(imagename)
    tclean(vis=vis,
           imagename=imagename,
           mask=mask,
           imsize=imc.imsize,
           cell=imc.cell,
           gridder='standard',
           restoringbeam='common',
           weighting='briggs',
           robust=imc.robust,
           specmode='mfs',
           interactive=True,
           niter=100000,
           savemodel='modelcolumn')
    tclean(vis=vis,
           imagename=imagename,
           imsize=imc.imsize,
           cell=imc.cell,
           gridder='standard',
           restoringbeam='common',
           weighting='briggs',
           robust=imc.robust,
           specmode='mfs',
           interactive=False,
           niter=0,
           calcres=False,
           calcpsf=False,
           savemodel='modelcolumn')
    # ^ NOTE The interactve thru calcpsf options are necessary to save the
    #        model
    print('-- Check RMS and beam')


def selfcal_pcal(imc, combine='spw,scan', trial='1'):
    """
    Self Calibration. Generate a per observation solution first.
    """
    if trial == '1':
        solint = 'inf'  # Sum entire scheduling block
    elif trial == '2':
        solint = '1800s'  # Half block
    else:
        raise ValueError('Invalid pcal trial: {0}'.format(trial))
    caltable = imc.path_base + '.pcal' + trial
    rmtables(caltable)
    gaincal(vis=imc.vislf,
            caltable=caltable,
            gaintype='T',
            refant=imc.refant,
            calmode='p',
            combine=combine,
            solint=solint,
            minsnr=3.0,
            minblperant=6)
    print('-- Check number of solutions lost')


def selfcal_pcal_plot(imc, trial='1'):
    """ Self Calibration. Check the solution """
    plotcal(caltable=imc.path_base+'.pcal'+trial,
            xaxis='time', yaxis='phase', timerange='', iteration='',
            subplot=111, plotrange=[0,0,-180,180])


def selfcal_apcal(imc, trial='1'):
    """ Self Calibration. Apply the calibration from the phase only solution """
    flagmanager(imc.vislf, mode='restore', versionname='startup')
    applycal(vis=imc.vislf,
             spwmap=np.zeros(54),
             interp='linearPDperobs',
             gaintable=[imc.path_base+'.pcal'+trial],
             calwt=False,
             flagbackup=False)


###############################################################################
# Line imaging
###############################################################################

def clean_line(imc, spw, contsub=False, fullcube=False, interactive=True,
               export=False, use_existing_psf=False, automask=False, **kwargs):
    """
    Use `tclean` to CLEAN a spectral cube. Additional keyword arguements are
    passed to `tclean`.

    Parameters
    ----------
    imc : ImageConfig
    spw : Spw
    contsub : bool
        Use the continuum subtracted measurement set
    fullcube : bool
        Yield a cube with the full number of channels, as opposed to clipping
        based on the `spw` around a velocity range about the line.
    interactive : bool
        Interactively clean in the viewer
    export : bool
        Export the CASA image files to FITS
    use_existing_psf : bool
        Skip first major cycle, using existing `.psf` file
    automask : str, default None
        Use automasking from:
            'auto'  -> usemask='autothresh'
            'multi' -> usemask='multithresh'
    """
    vis = imc.viscs if contsub else imc.vis
    imagename = '{0}_{1}'.format(imc.path_base, spw.name)
    tclean_kwargs = {
        'interactive': interactive,
        'niter': 1000000 if interactive else 0,
        'start': '' if fullcube else imc.start_vlsr(spw),
        'width': '' if fullcube else spw.velo_width,
        'nchan': -1 if fullcube else spw.nchan,
        'calcpsf': not use_existing_psf,
    }
    if automask == 'auto':
        kwargs.update({
            'usemask': 'auto-thresh' if automask else None,
            'maskthreshold': '33mJy' if automask else None,
            'maskresolution': 1.0 if automask else None,
        })
    elif automask == 'multi':
        kwargs.update({
            'usemask': 'auto-multithresh' if automask else None,
            'noisethreshold': 4.0 if automask else None,
            'lownoisethreshold': 1.5 if automask else None,
            'smoothfactor': 1.0 if automask else None,
            'minbeamfrac': 0.3 if automask else None,
            'growiterations': 4 if automask else None,
        })
    if kwargs is not None:
        tclean_kwargs.update(kwargs)
    print(':: Running clean')
    tclean(vis=vis,
           imagename=imagename,
           field='0',
           spw=imc.get_one_spw(spw, contsub=contsub),
           specmode='cube',
           outframe='lsrk',
           veltype='radio',
           restfreq=spw.restfreq,
           imsize=imc.imsize,
           cell=imc.cell,
           gridder=imc.gridder,
           deconvolver=imc.deconvolver,
           scales=imc.scales,
           smallscalebias=imc.smallscalebias,
           restoringbeam='common',
           weighting='briggs',
           robust=imc.robust,
           **tclean_kwargs)
    if export:
        export_fits(imagename+'.image')


def make_line_psf(imc, spw=None):
    if spw is None:
        for line_name, spw in SPWS.iteritems():
            print(':: Molecule {0}'.format(line_name))
            clean_line(imc, spw, interactive=False)
    else:
        clean_line(imc, spw, interactive=False)


def clean_existing_line(imc, spw, **kwargs):
    clean_line(imc, spw, use_existing_psf=True)


def make_cont_sub(imc):
    uvcontsub(vis=imc.vis,
              field='0',
              fitspw=imc.line_chans,
              excludechans=True,
              combine='spw',
              solint='int',
              fitorder=0)


###############################################################################
# Moment analysis
###############################################################################

def make_im_moments(imc, spw=None, chans=''):
    #for name in SPWS.keys():
    imagename = imc.path_base + '_{0}.image'.format(spw.name)
    outname = '{base}/moments/{line}_snr{snr}'
    rms = 2.0e-3 if spw.velo_width == '0.68km/s' else 2.8e-3  # in mJy/beam
    # FIXME `includepix`
    immoments(imagename=imagename,
              moments=[0],
              excludepix=[-100,1.0e-2],
              axis='spectral',
              chans=chans)
    immoments(imagename=imagename,
              moments=[-1, 1, 2, 8],
              includepix=[2*rms, 1],
              excludepix=[-100,1.0e-2],
              axis='spectral',
              chans=chans)


def make_ia_moments(imc, spw=None):
    if spw is None:
        spwl = SPWS.values()
    else:
        spwl = [spw]
    infile_fmt = imc.path_base + '_{0}.image'
    outfile_fmt = imc.name + '/moments/{line}_snr{snr}_smooth{smooth}'
    maxv = 100
    smoothaxes = [0, 1]
    smoothtypes = ['gauss', 'gauss']
    smoothwidths = [30, 30]  # FWHM of kernel in pixels, 0.1 arcsec
    for spw in spwl:
        if spw.velo_width == '0.68km/s':
            rms = 2.0e-3  # Jy/beam
        else:
            rms = 2.8e-3
        ia.open(infile_fmt.format(spw.name))
        for snr in (2, 3, 4):
            # unsmoothed moments
            ia.moments(
                moments=[-1,0,1,2,8],
                axis=3,
                includepix=[snr*rms,maxv],
                outfile=outfile_fmt.format(line=spw.name, snr=str(snr), smooth='0'),
                overwrite=True,
            ).done()
            # smoothed moments
            ia.moments(
                moments=[-1,0,1,2,8],
                axis=3,
                includepix=[snr*rms/4.44,maxv],
                smoothaxes=smoothaxes,
                smoothtypes=smoothtypes,
                smoothwidths=smoothwidths,
                outfile=outfile_fmt.format(line=spw.name, snr=str(snr),
                    smooth=str(smoothwidths[0])),
                overwrite=True,
            ).done()
        ia.close()


def pbcor_moments(imc, spw=None, overwrite=True):
    if spw is None:
        spwl = SPWS.values()
    else:
        spwl = [spw]
    outfile_fmt = imc.name + '/moments/{line}_snr{snr}_smooth{smooth}'
    print(':: Primary beam correcting images')
    for spw in spwl:
        print('-- {0}'.format(spw.name))
        pbimage = '{0}_{1}.pb'.format(imc.path_base, spw.name)
        # NOTE We need to remove spectral axis in the primary beam image in
        # order for both data to have the same shape.  Luckily, `impbcor` can
        # be fed an array as well as a file name.
        ia.open(pbimage)
        pbdata = ia.getregion()[...,0]
        ia.close()
        infiles = glob.glob('{0}/moments/{1}_*'.format(imc.name, spw.name))
        infiles = [s for s in infiles
            if s.endswith('average')
            or s.endswith('integrated')
            or s.endswith('maximum')]
        for imagename in infiles:
            impbcor(imagename=imagename, pbimage=pbdata,
                    outfile=imagename+'.pbcor', overwrite=overwrite)


def export_moments(imc):
    infiles = glob.glob('{0}/moments/*'.format(imc.name))
    for imagename in infiles:
        export_fits(imagename)


###############################################################################
# Diagnostic plots
###############################################################################

def plot_find_continuum(imc, usemask=False):
    outdir = '{0}/plots/findcont/'.format(imc.name)
    mask = imc.path_base + '.mask' if usemask else ''
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    do_spws = ['dcop', 'sio', 'p-h2co_303_202', 'p-h2co_322_221',
               'p-h2co_321_220', 'c18o', 'co', 'n2dp']
    for spw_str in do_spws:
        spw = SPWS[spw_str]
        print(':: Plotting: {0}'.format(spw.name))
        imagename = imc.path_base + '_{0}.image'.format(spw.name)
        outname = outdir + '{0}_{1}.png'.format(imc.name, spw.name)
        findContinuum.findContinuum(img=imagename, mask=mask, png=outname,
                                    overwrite=True)


