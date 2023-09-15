"""This is a python port of Justin Braaten's msslib.

The original msslib can be found here: https://github.com/gee-community/msslib
This port was written by Rylan Boothman in December 2021.

The intent of the original msslib is "to make it easy to work with Landsat MSS
data in Earth Engine." The intent of this port to python is to make it easier
to work with Landsat MSS images when using the Earth Engine python api.

You should import the earthengine python api and authenticate and initialize it
before importing the msslib.

For example:
    ```
    import ee
    ee.Authenticate()
    ee.Initialize()

    import msslib
    ```

Copyright 2020 Justin Braaten

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ee
import math
from IPython.display import Image, display

# A dictionary of false color visualization parameters for MSS DN images.
visDn = {
    'bands': ['nir', 'red', 'green'],
    'min': [47, 20, 27],
    'max': [142, 92, 71],
    'gamma': [1.2, 1.2, 1.2]
}

visDn2 = {
    'bands': ['nir', 'red_edge', 'red'],
    'min': [47, 20, 20],
    'max': [142, 142, 92],
    'gamma': [1.2, 1.2, 1.2]
}

# A dictionary of false color visualization parameters for MSS radiance images.
visRad = {
    'bands': ['nir', 'red', 'green'],
    'min': [23, 15, 25],
    'max': [67, 62, 64],
    'gamma': [1.2, 1.2, 1.2]
}

# A dictionary of false color visualization parameters for MSS TOA reflectance
# images.
visToa = {
    'bands': ['nir', 'red', 'green'],
    'min': [0.0896, 0.0322, 0.0464],
    'max': [0.2627, 0.1335, 0.1177],
    'gamma': [1.2, 1.2, 1.2]
}

visToa2 = {
    'bands': ['nir', 'red_edge', 'red'],
    'min': [0.0896, 0.0322, 0.0322],
    'max': [0.2627, 0.2627, 0.1335],
    'gamma': [1.2, 1.2, 1.2]
}

# A dictionary of visualization parameters for MSS NDVI images.
visNdvi = {
    'bands': ['ndvi'], 'min': 0.1, 'max': 0.8
}

_BAND_NAMES = ['green', 'red', 'red_edge', 'nir', 'QA_PIXEL', 'QA_RADSAT']

##############################################################################
# FETCH COLLECTIONS
##############################################################################


def getWrs1GranuleGeom(granuleId):
    """Gets the geometry for a given WRS-1 granule.

    Only returns results for granules that intersect land on the descending
    path.

    Args:
        granuleId: String containing the PPPRRR granule ID.

    Returns:
        Dictionary with three elements: 'granule' a ee.Feature, 'centroid' a
        ee.Geometry, and 'bounds' a ee.Geometry with a 40 km buffer.
    """
    granule = ee.Feature(
        ee.FeatureCollection('users/jstnbraaten/wrs/wrs1_descending_land')
        .filter(ee.Filter.eq('PR', granuleId)).first())
    centroid = granule.centroid(300).geometry(300)
    bounds = granule.geometry(300).buffer(40000)
    return ee.Dictionary({
        'granule': granule,
        'centroid': centroid,
        'bounds': bounds
    })


def getPr(img):
    path = img.getNumber('WRS_PATH').format('%03d')
    row = img.getNumber('WRS_ROW').format('%03d')
    return path.cat(row)


def _filterById(id, col):
    """Excludes an image from a collection by Image ID.

    Used as the algorithm input in the msslib.filterById() method

    Args:
        id: The image id to filter out of the image collection, given as the
            value of the image's LANDSAT_SCENE_ID property.
        col: The image collection to filter.

    Returns:
        The filtered image collection.
    """
    return ee.ImageCollection(col).filter(
        ee.Filter.neq('LANDSAT_SCENE_ID', ee.String(id))
    )


def filterById(col, imgList):
    """Excludes a list of images from a collection by Image ID.

    Used in the msslib.filterCol() method.

    Args:
        col: An ee.ImageCollection to filter.
        imgList: A list of image IDs to filter from the collection, given as
            the image's system:index property.

    Returns:
        The filtered image collection.
    """
    return ee.ImageCollection(ee.List(imgList).iterate(_filterById, col))


def _getById(imageID, col):
    """Returns the image in col whose ID matches image ID.

    Uses the LANDSAT_SCENE_ID metadata property as image ID. If no image in
    col has the matching ID, returns None.

    Args:
        imageID: A string containing the LANDSAT_SCENE_ID to get.
        col: An ee.ImageCollection to filter

    Returns:
        The image in col with the matching imageID, if such an image exists.
    """
    filt = ee.Filter.eq('LANDSAT_SCENE_ID', imageID)
    return ee.ImageCollection(col).filter(filt).first()


def getById(col, imgList):
    """Excludes all images in a collection whose id is NOT in imgList

    Used in the msslib.filterCol() method.

    Args:
        col: an ee.ImageCollection to filter.
        imgList: A list of all image IDs to keep in the collection, given as
            image's system:index property.

    Returns:
        The filtered image collection.
    """
    return ee.ImageCollection(ee.List(imgList).map(
        lambda imageId: _getById(imageId, col), dropNulls=True
    ))


def filterCol(col, aoi, maxRmseVerify, maxCloudCover, yearRange, doyRange,
              excludeIds, includeIds, wrs):
    """Filters an MSS image collection by bounds, date, and quality properties

    By default, it excludes images that do not have all four reflectance bands
    present and/or are only processed to level L1G. It is intended to handle
    only one MSS collection at a time i.e. no merged collections. Used by the
    msslib.getCol() method.

    Args:
        col: The ee.ImageCollection to filter.
        aoi: ee.Geometry() pased to filterBounds on col.
        maxRmseVerify: float used to filter images on GEOMETRIC_RMSE_VERIFY
            property, images whose property value is less than or equal to the
            given value are kept.
        maxCloudCover: A float used to filter images on the CLOUD_COVER
            property, images whose property value is less than or equal to the
            given value are kept.
        yearRange: A 2-tuple containing the start year and end year inclusive
            for which images will be returned.
        doyRange: A 2-tuple contianing the start day of year and end day of
            year for which images will be returned.
        excludeIds: list of image IDs to be excluded using msslib.filterById.
        includeIds: list of image IDs that can be included the output
            collection, passed to msslib.getById.
        wrs: A string (either 'wrs1' or 'wrs2') indicating whether the image
            collection contains WRS-1 or WRS-2 images.

    Returns:
        The filtered image collection.
    """
    bandsPresent = {
        'wrs1': [
            'PRESENT_BAND_4', 'PRESENT_BAND_5',
            'PRESENT_BAND_6', 'PRESENT_BAND_7'
        ],
        'wrs2': [
            'PRESENT_BAND_1', 'PRESENT_BAND_2',
            'PRESENT_BAND_3', 'PRESENT_BAND_4'
        ]
    }

    if aoi is not None:
        col = col.filterBounds(aoi)

    col = col.filter(ee.Filter.neq('DATA_TYPE', 'L1G')) \
        .filter(ee.Filter.eq(bandsPresent[wrs][0], 'Y')) \
        .filter(ee.Filter.eq(bandsPresent[wrs][1], 'Y')) \
        .filter(ee.Filter.eq(bandsPresent[wrs][2], 'Y')) \
        .filter(ee.Filter.eq(bandsPresent[wrs][3], 'Y')) \
        .filter(ee.Filter.lte('GEOMETRIC_RMSE_VERIFY', maxRmseVerify)) \
        .filter(ee.Filter.lte('CLOUD_COVER', maxCloudCover))

    if yearRange is not None:
        col = col.filter(ee.Filter.calendarRange(
            yearRange[0], yearRange[1], 'year'
        ))

    if doyRange is not None:
        col = col.filter(ee.Filter.calendarRange(
            doyRange[0], doyRange[1], 'day_of_year'
        ))

    if excludeIds is not None:
        col = filterById(col, excludeIds)

    if includeIds is not None:
        col = getById(col, includeIds)

    return col


def getCol(aoi=None, maxRmseVerify=0.5, maxCloudCover=50, wrs='1&2',
           yearRange=[1972, 2000], doyRange=[1, 365], excludeIds=None,
           includeIds=None):
    """Assembles a Landsat MSS image collection.

    Includes images from USGS Collection 2 T1 and T2 acquired by satellites
    1-5. Removes L1G images and images without a complete set of reflectance
    bands. Additional default and optional filtering criteria are applied,
    including by bounds, geometric error, cloud cover, year, and day of year.
    All image bands are named consistently: ['green', 'red', 'red_edge', 'nir',
    QA_PIXEL']. Adds 'wrs' property to all images designating them as 'WRS-1' or
    'WRS-2'.

    Args:
        aoi: ee.Geometry() pased to filterBounds on col.
        maxRmseVerify: float used to filter images on GEOMETRIC_RMSE_VERIFY
            property, images whose property value is less than or equal to the
            given value are kept.
        maxCloudCover: A float used to filter images on the CLOUD_COVER
            property, images whose property value is less than or equal to the
            given value are kept.
        wrs: A string indicating what World Reference System types to allow
            in the collection. If '1' is in the string, WRS-1 images
            will be included. If '2' is in the string, WRS-2 images
            will be include. If both '1' and '2' are in the string, both WRS-1
            and WRS-2 images will be included.
        yearRange: A 2-tuple containing the start year and end year inclusive
            for which images will be returned.
        doyRange: A 2-tuple contianing the start day of year and end day of
            year for which images will be returned.
        excludeIds: list of image IDs to be excluded using msslib.filterById.
        includeIds: list of image IDS to be included using msslib.getById.

    Returns:
        An ee.ImageCollection
    """
    wrs1Col = ee.ImageCollection([])
    wrs2Col = ee.ImageCollection([])

    params = {
        'aoi': aoi, 'maxRmseVerify': maxRmseVerify,
        'maxCloudCover': maxCloudCover, 'yearRange': yearRange,
        'doyRange': doyRange, 'excludeIds': excludeIds,
        'includeIds': includeIds
    }

    def _wrs1_filter(x): return filterCol(x, wrs='wrs1', **params)
    def _wrs2_filter(x): return filterCol(x, wrs='wrs2', **params)

    # gather and filter all WRS-1 images
    if '1' in wrs:
        mss1T1 = _wrs1_filter(ee.ImageCollection('LANDSAT/LM01/C02/T1'))
        mss1T2 = _wrs1_filter(ee.ImageCollection('LANDSAT/LM01/C02/T2'))
        mss2T1 = _wrs1_filter(ee.ImageCollection('LANDSAT/LM02/C02/T1'))
        mss2T2 = _wrs1_filter(ee.ImageCollection('LANDSAT/LM02/C02/T2'))
        mss3T1 = _wrs1_filter(ee.ImageCollection('LANDSAT/LM03/C02/T1'))
        mss3T2 = _wrs1_filter(ee.ImageCollection('LANDSAT/LM03/C02/T2'))

        wrs1Col = mss1T1.merge(mss1T2).merge(mss2T1).merge(mss2T2) \
            .merge(mss3T1).merge(mss3T2) \
            .map(lambda img: img.rename(_BAND_NAMES).set('wrs', 'WRS-1'))

    # gather and filter all WRS-2 images
    if '2' in wrs:
        mss4T1 = _wrs2_filter(ee.ImageCollection('LANDSAT/LM04/C02/T1'))
        mss4T2 = _wrs2_filter(ee.ImageCollection('LANDSAT/LM04/C02/T2'))
        mss5T1 = _wrs2_filter(ee.ImageCollection('LANDSAT/LM05/C02/T1'))
        mss5T2 = _wrs2_filter(ee.ImageCollection('LANDSAT/LM05/C02/T2'))

        wrs2Col = mss4T1.merge(mss4T2).merge(mss5T1).merge(mss5T2) \
            .map(lambda img: img.rename(_BAND_NAMES).set('wrs', 'WRS-2'))

    # merge and sort by time the WRS-1 and WRS-2 collections
    return wrs1Col.merge(wrs2Col).map(lambda img: (
        img.set({
            'start_doy': doyRange[0],
            'end_doy': doyRange[1],
            'year': img.date().get('year'),
            'doy': img.date().getRelative('day', 'year'),
        })
    )).sort('system:time_start')


def process(img):
    """ Process an image to be consistent with images returned by getCol.

    Renames bands to ['green', 'red', 'red_edge', 'nir', 'QA_PIXEL']
    Adds the following properties: ['wrs', 'year', 'doy']

    Any downstream function expecting an image output by getCol should work
    with an image output by this function as long as the function does not need
    to work with the 'start_doy', and 'end_doy' properties as there is no for
    them equivalent that can be added by this method

    Args:
        img: MSS ee.Image

    Returns:
        ee.Image
    """
    img = img.rename(_BAND_NAMES)
    return img.set({
        'doy': img.date().getRelative('day', 'year'),
        'year': img.date().get('year'),
        'wrs': ee.String('WRS-').cat(img.getNumber('WRS_TYPE').format('%d')),
    })



##############################################################################
# IMAGE ASSESSMENT
##############################################################################

def viewThumbnails(col, visParams=None, thumbnailParams=None,
                   preprocess=None, sort=False):
    """Prints image collection thumbnails with accompanying image IDs.

    Useful for quickly evaluating a collection. The image IDs can be recorded
    and used as entries in the excludeIds argument to msslib.getCol().

    Args:
        col: The ee.ImageCollection to be displayed, should originate
            from msslib.getCol().
        visParams: A dictionary of visualization parameters passed to
            ee.Image.visualize().
        thumbnailParams: A dictionaray passed to ee.Image.getThumbUrl().
        preprocess: A function that takes an ee.Image and returns an ee.Image
            and calculates toa, adds bands etc. to make the image ready for
            visualization.
        sort: If true, sort by system:time_start before displaying

    Returns:
        None
    """
    print('Please wait patiently, images may not load immediately')

    if visParams is None:
        visParams = visDn

    if thumbnailParams is None:
        thumbnailParams = {}
    if 'dimensions' not in thumbnailParams:
        thumbnailParams['dimensions'] = 512
    if 'crs' not in thumbnailParams:
        thumbnailParams['crs'] = 'EPSG:3857'

    if preprocess is None:
        preprocess = calcToa

    if sort:
        col = col.sort('system:time_start')

    imgList = col.toList(col.size())

    for i in imgList.getInfo():
        img = ee.Image(i['id']).rename(_BAND_NAMES)

        print(img.get('LANDSAT_SCENE_ID').getInfo())

        img = preprocess(img)
        img = img.unmask(0).visualize(**visParams)
        display(Image(url=img.getThumbUrl(thumbnailParams)))


##############################################################################
# IMAGE MANIPULATION
##############################################################################


def scaleDn(img, unit):
    """Converts DN values to either radiance or TOA reflectance.

    Args:
        img: An ee.Image originating from msslib.getCol().
        unit: A string indicating whether to convert DN to units of radiance
            ('radiance') or TOA reflectance ('reflectance').

    Returns:
        An ee.Image converted to the new unit type.
    """
    mult = 'REFLECTANCE_MULT_BAND'
    add = 'REFLECTANCE_ADD_BAND'

    if unit == 'radiance':
        mult = 'RADIANCE_MULT_BAND'
        add = 'RADIANCE_ADD_BAND'

    gainBands = ee.List(img.propertyNames()) \
        .filter(ee.Filter.stringContains('item', mult)) \
        .sort()
    biasBands = ee.List(img.propertyNames()) \
        .filter(ee.Filter.stringContains('item', add)) \
        .sort()

    gainImg = ee.Image.cat(
        ee.Image.constant(img.get(gainBands.getString(0))),
        ee.Image.constant(img.get(gainBands.getString(1))),
        ee.Image.constant(img.get(gainBands.getString(2))),
        ee.Image.constant(img.get(gainBands.getString(3)))).toFloat()

    biasImg = ee.Image.cat(
        ee.Image.constant(img.get(biasBands.getString(0))),
        ee.Image.constant(img.get(biasBands.getString(1))),
        ee.Image.constant(img.get(biasBands.getString(2))),
        ee.Image.constant(img.get(biasBands.getString(3)))).toFloat()

    dnImg = img.select([0, 1, 2, 3]).multiply(gainImg).add(biasImg).toFloat()

    # TODO: can get around needing to copy properties and add the QA band
    # by addBands with overwrite: img.addBands(dnImg, None, True)

    return ee.Image(dnImg
        .addBands(img.select('QA_PIXEL'))
        .copyProperties(img, img.propertyNames()))


def calcRad(img):
    """Converts DN values to radiance.

    Args:
        img: An ee.Image.

    Returns:
        The image in radiance units.
    """
    return scaleDn(img, 'radiance')


def calcToa(img):
    """Converts DN values to radiance.

    Args:
        img: An ee.Image.

    Returns:
        The image in TOA reflectance units.
    """
    return scaleDn(img, 'reflectance')


def addNdvi(img):
    """Adds NDVI transformation as a band ('ndvi') to the input image.

    Args:
        img: An ee.Image.

    Returns:
        The input image with a new band containing the NDVI and named 'ndvi'.
    """
    ndvi = img.normalizedDifference(['nir', 'red']).rename('ndvi')
    return ee.Image(
        img.addBands(ndvi).copyProperties(img, img.propertyNames()))


def addTc(img):
    """Adds Tasseled Cap indices to the input image.

    Adds Tasseled Cap brightness ('tcb'), greenness ('tcg'), yellowness ('tcy')
    and angle ('tca') to the input image. See Kauth and Thomas, 1976.
    https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1160&context=lars_symp

    Args:
        img: An ee.Image.

    Returns:
        The input image with new bands tcb, tcg, tcy, and tca.
    """
    bands = img.select([0, 1, 2, 3])
    tcbCoeffs = ee.Image.constant([0.433, 0.632, 0.586, 0.264])
    tcgCoeffs = ee.Image.constant([-0.290, -0.562, 0.600, 0.491])
    tcyCoeffs = ee.Image.constant([-0.829, 0.522, -0.039, 0.194])
    tcb = bands.multiply(tcbCoeffs).reduce(ee.Reducer.sum()).toFloat()
    tcg = bands.multiply(tcgCoeffs).reduce(ee.Reducer.sum()).toFloat()
    tcy = bands.multiply(tcyCoeffs).reduce(ee.Reducer.sum()).toFloat()
    tca = (tcg.divide(tcb)).atan().multiply(180 / math.pi).toFloat()
    tc = ee.Image.cat(tcb, tcg, tcy, tca).rename('tcb', 'tcg', 'tcy', 'tca')
    return ee.Image(img.addBands(tc).copyProperties(img, img.propertyNames()))


def getQaMask(img):
    """Gets the BQA quality band as boolean layer.

    1 indicates a good pixel and 0 indicates a bad pixel.

    Args:
        img: An ee.Image originating from msslib.getCol().

    Returns:
        A image with one boolean band named BQA_mask.
    """
    # return img.select('BQA').eq(32).rename('BQA_mask')
    qaPixelMask = img.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    qaRadsatMask = img.select('QA_RADSAT').eq(0)
    return qaPixelMask.updateMask(qaRadsatMask).rename('QA_mask')


def addQaMask(img):
    """Adds the mask created by getQaMask() to the input image as a new band.

    Args:
        img: An ee.Image originating from msslib.getCol().

    Returns:
        The input image with a new band named BQA_mask.
    """
    return img.addBands(getQaMask(img))


def applyQaMask(img):
    """Applies the BQA quality band to an image as mask.

    It masks out cloud pixels and those exhibiting radiometric saturation, as
    well as pixels associated with missing data. Cloud identification is
    limited to mostly thick cumulus clouds; note that snow and very bright
    surface features are often mislabeled as cloud. Radiometric saturation in
    MSS images usually manifests as entire or partial image pixel rows being
    highly biased toward high values in a single band, when when visualized,
    can appear as tinted red, green, or blue.

    Args:
        img: An ee.Image originating from msslib.getCol().

    Returns:
        The input image after applying the BQA mask.
    """
    return img.updateMask(getQaMask(img))


###############################################################################
# MSSCVM
###############################################################################


def cloudLayer(img):
    """Returns the MSScvm cloud layer.

    Args:
        img: An MSS TOA ee.Image originating from msslib.getCol() and
            msslib.calcToa().

    Returns:
        An image with one band named 'clouds' that is one where there are
        clouds and zero otherwise.
    """
    cloudPixels = img.normalizedDifference(['green', 'red']) \
        .gt(0) \
        .multiply(img.select('green').gt(0.175)) \
        .add(img.select('green').gt(0.39)) \
        .gt(0)

    # nine-pixel minimum connected component sieve
    cloudPixels = cloudPixels.selfMask() \
        .connectedPixelCount(10, True) \
        .reproject(img.projection()) \
        .gte(0) \
        .unmask(0) \
        .rename('cloudtest')

    kernel = ee.Kernel.circle(radius=2, units='pixels', normalize=True)

    # two pixel buffer, eight neighbor rule
    return cloudPixels.focal_max(radius=2, kernel=kernel) \
        .reproject(img.projection()) \
        .rename('clouds')


def dilateZeroOne(img, pixel_distance, ground_distance):
    """Dilates a boolean image.

    Expands all regions in the image with value 1 outwards by ground_distance
    meters. Not part of Justin Braaten's original msslib, used here when
    calculating the water layer.

    Args:
        img: An ee.Image, should be all zeros and ones.
        pixel_distance: An integer passed to fastDistanceTransform
        ground_distance: An integer indicating how far to dilate the image in
            meters.

    Returns:
        The input image after applying the dilation.
    """
    img = img.clip(img.geometry())
    d = img.fastDistanceTransform(pixel_distance).sqrt()
    d = d.multiply(ee.Image.pixelArea().sqrt())
    return d.lt(ground_distance)

def waterLayerJB(img):
    """Justin Braaten's version of the water layer
    See docstring for waterLayer method for explanation of differences

    Args:
        img: An ee.image originating from msslib.getCol() and msslib.calcToa().

    Returns:
        An ee.Image with one band called 'water' that is 1 where there is water
        and zero otherwise.
    """
    # threshold on NDVI
    mssWater = img.normalizedDifference(['nir', 'red']).lt(-0.085)

    # Get max extent of water 1985-2018
    waterExtent = ee.Image('JRC/GSW_1/GlobalSurfaceWater') \
        .select('max_extent')

    # Get intersection of MSS water and max extent.
    return mssWater.multiply(waterExtent) \
        .reproject(img.projection()) \
        .rename('water')


def waterLayer(img):
    """Returns the MSScvm water layer.

    Unlike in Justin Braaten's orginal msslib, the water layer is buffered
    outward by 80 meters here using dilateZeroOne.

    The threshold applied to the NDVI to determine if the JRC Global Surface
    Water dataset is accurate in the image is changed from Justin Braaten's
    original value of -0.085 to 0.2

    Args:
        img: An ee.Image orgination from msslib.getCol() and msslib.calcToa().

    Returns:
        An ee.Image with one band called 'water' that is 1 where there is water
        and zero otherwise.
    """
    # store geometry to clip after calling lt with a constant
    geom = img.geometry()

    # threshold on NDVI
    mssWater = img.normalizedDifference(['nir', 'red']).lt(0.2)
    mssWater = mssWater.clip(geom)

    # get the max extent of water 1985-2018
    waterExtent = ee.Image('JRC/GSW1_1/GlobalSurfaceWater') \
        .select('max_extent')

    # get intersection of MSS water and max extent
    actual_water = mssWater.multiply(waterExtent) \
        .reproject(img.projection()) \
        .rename('water')

    # buffer water slightly
    buffered_water = dilateZeroOne(actual_water, 40, 80)
    return buffered_water.reproject(img.projection())


def getDem(img):
    """Assembles a global DEM in the projection of the input image.

    Uses a mosaic of JAXA/ALOS/AW3d30/V2_2 and USGS/GMTED2010

    Args:
        img: An ee.Image.

    Returns:
        An ee.Image containing the global DEM.
    """
    aw3d30 = ee.Image('JAXA/ALOS/AW3D30/V2_2').select('AVE_DSM').rename('elev')
    GMTED2010 = ee.Image('USGS/GMTED2010').rename('elev')
    return ee.ImageCollection([GMTED2010, aw3d30]) \
        .mosaic() \
        .reproject(img.projection())


def radians(img):
    return img.toFloat().multiply(math.pi).divide(180)


def getIll(img, slope, aspect):
    """Returns a terrain illumination image.

    Args:
        img: An ee.Image originating from msslib.getCol() and msslib.calcToa()
        slope: A terrain slope image in units of degrees.
        aspect: A terrain aspect image in units of degrees.

    Returns:
        An ee.Image.
    """
    # get sun info
    azimuth = img.get('SUN_AZIMUTH')
    zenith = ee.Number(90).subtract(img.getNumber('SUN_ELEVATION'))

    # convert slope and aspect degrees to radians
    slopeRad = radians(slope)
    aspectRad = radians(aspect)

    # calculate illumination
    azimuthImg = radians(ee.Image.constant(azimuth))
    zenithImg = radians(ee.Image.constant(zenith))
    left = zenithImg.cos().multiply(slopeRad.cos())
    right = zenithImg.sin() \
        .multiply(slopeRad.sin()) \
        .multiply(azimuthImg.subtract(aspectRad).cos())
    return left.add(right)


def topoCorrB4(img, dem):
    """Returns MSS NIR TOA reflectance band corrected for topography.

    Corrects for topography via Minnaert correction.

    Args:
        img: An ee.Image origination from msslib.getCol() and msslib.calcToa()
        dem: An ee.Image digital elevation model.

    Returns:
        An ee.Image.
    """
    terrain = ee.Algorithms.Terrain(dem)
    slope = terrain.select(['slope'])
    aspect = terrain.select(['aspect'])

    # get k image, define polynomial coefficients to calculate Minnaert
    # value as function of slope.
    # Ge, H., Lu. D., He, S., Xu, A., Zhou, G., & Du, H. (2008). Pixel-based
    # Minnaert correction method for reducting topographic effects on a Landsat
    # 7 ETM+ image. Photogrammetric Engineering & Remote Sensing, 75(11),
    # 1343-1350.
    kImg = slope.resample('bilinear') \
        .where(slope.gt(50), 50) \
        .polynomial([
            1.0021313684, -0.1308793751, 0.0106861276, -0.0004051135,
            0.0000071825, -4.88e-8
        ])

    ill = getIll(img, slope, aspect)

    cosTheta = radians(ee.Image.constant(ee.Number(90).subtract(
        ee.Number(img.get('SUN_ELEVATION'))))).cos()

    correction = (cosTheta.divide(ill)).pow(kImg)
    return img.select('nir').multiply(correction)


def shadowLayer(img, dem, clouds, maxDist=50):
    """Returns the MSScvm shafow layer.

    Args:
        img: An ee.Image originating from msslib.getCol() and msslib.calcToa()
        dem: An ee.Image digital elevation model
        clouds: An ee.Image originating from msslib.cloudLayer().
        maxDist: int, the maximum distance in pixels away from the cloud layer
            to search for cloud shadow, passed to directionalDistanceTransform

    Returns:
        An ee.Image with one band named shadow that is one where there is
        cloud shadow and zero otherwise.
    """
    # correct B4 reflectance for topography
    b4c = topoCorrB4(img, dem)

    # threshold B$ - target dark pixels.
    # make this true for all pixels to use full cloud projection.
    shadows = b4c.lt(0.11)

    # project clouds as potential shadow.
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(img.get('SUN_AZIMUTH'))
    )
    cloudProj = clouds.directionalDistanceTransform(shadow_azimuth, maxDist) \
        .reproject(crs=img.projection(), scale=60) \
        .select('distance') \
        .gt(0) \
        .unmask(0)

    water = waterLayer(img)

    # exclude water pixels from intersection of cloud projection and dark
    # pixels.
    return shadows.multiply(water.Not()) \
        .multiply(cloudProj) \
        .focal_max(2) \
        .reproject(img.projection()) \
        .rename('shadow')


def addMsscvm(img, shadowSearchDist=50):
    """Adds the MSScvm band to the input image.

    The new band will be named 'msscvm'. Values of 0 designate pixels as clear,
    1 as clouds, and 2 as shadows.

    Args:
        img: An ee.Image originating from msslib.getCol() and msslib.calcToa()
        shadowSearchDist: int, the maximum number of pixels away from cloud to
            search for cloud shadow

    Returns:
        The input ee.Image with new band named msscvm.
    """
    dem = getDem(img)
    clouds = cloudLayer(img).selfMask()
    shadows = shadowLayer(img, dem, clouds, shadowSearchDist).selfMask().add(1)
    return img.addBands(shadows.blend(clouds).unmask(0).rename('msscvm'))


def applyMsscvm(img, shadowSearchDist=50):
    """Applies the MSScvm mask to the input image.

    Pixels identified as cloud or cloud shadow in the MSScvm will be masked.

    Args:
        img: An ee.Image originating from msslib.getCol() and msslib.calcToa()
        showdowSearchDist: int, the maximum number of pixels away from cloud to
            search for cloud shadow

    Returns:
        The input ee.Image with clouds and cloud shadow masked.
    """
    dem = getDem(img)
    clouds = cloudLayer(img)
    shadows = shadowLayer(img, dem, clouds, shadowSearchDist)
    mask = clouds.add(shadows).eq(0)
    return img.updateMask(mask)
