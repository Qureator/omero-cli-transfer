import os
from typing import BinaryIO, Union, Dict, Optional, Tuple

import numpy as np
import ome_types
import tifffile
from ome_types import model as ome_model
from ome_types.model import Channel, Image, Pixels, simple_types
from ome_types.model.pixels import DimensionOrder
from omero.gateway import ImageWrapper
from omero.util.pixelstypetopython import toNumpy as pixel_type_to_numpy
from omero_acquisition_transfer.transfer.pack import export_image_metadata

from .ome_tiff import write_ome_tiff

__all__ = ["export_as_ome_tiff", "export_ome_metadata"]


ometypedict: Dict[np.dtype, simple_types.PixelType] = {
    np.dtype(np.int8): simple_types.PixelType.INT8,
    np.dtype(np.int16): simple_types.PixelType.INT16,
    np.dtype(np.int32): simple_types.PixelType.INT32,
    np.dtype(np.uint8): simple_types.PixelType.UINT8,
    np.dtype(np.uint16): simple_types.PixelType.UINT16,
    np.dtype(np.uint32): simple_types.PixelType.UINT32,
    np.dtype(np.float32): simple_types.PixelType.FLOAT,
    np.dtype(np.float64): simple_types.PixelType.DOUBLE,
    np.dtype(np.complex64): simple_types.PixelType.COMPLEXFLOAT,
    np.dtype(np.complex128): simple_types.PixelType.COMPLEXDOUBLE,
}


def _get_pixel_type(npdtype: np.dtype) -> simple_types.PixelType:
    ptype = ometypedict.get(npdtype)
    if ptype is None:
        raise ValueError("OMEXML get_pixel_type unknown type: " + npdtype.name)
    return ptype


def _gen_ome_metadata(array: np.ndarray) -> ome_types.OME:
    assert array.ndim == 5, f"array dimension should be 5, but {array.ndim}"

    # The dimension order of the given ndarray is assumed to be (T, C, Z, Y, X)
    t, c, z, y, x = array.shape

    channels = []
    for i in range(c):
        channel = Channel(
            id=f"Channel:{i + 1}",
            name=f"Channel:{i + 1}",
            samples_per_pixel=1,
        )
        channels.append(channel)

    image = Image(
        id=simple_types.ImageID("Image:0"),
        name="IMAGE",
        pixels=Pixels(
            id="Pixels:0",
            size_t=simple_types.PositiveInt(t),
            size_c=simple_types.PositiveInt(c),
            size_z=simple_types.PositiveInt(z),
            size_y=simple_types.PositiveInt(y),
            size_x=simple_types.PositiveInt(x),
            type=_get_pixel_type(array.dtype),
            dimension_order=DimensionOrder.XYZCT,
            channels=channels,
            metadata_only=True,
        ),
    )

    ome = ome_types.OME(images=[image])

    return ome


def read_ome_tiff(
    filename: Union[str, os.PathLike, BinaryIO, tifffile.FileHandle],
) -> Tuple[np.ndarray, ome_types.OME]:
    """Read a OME TIFF file, and return an image pixel data and ome metadata.

    Parameters
    ----------
    filename : Union[str, os.PathLike, BinaryIO, tifffile.FileHandle]
        File name or writable binary stream, such as an open file or BytesIO.

    Returns
    -------
    Tuple[np.ndarray, ome_types.OME]
        a tuple of (a 5D numpy array, an OME metadata object)
        The 5D numpy array is in the format: (T, C, Z, Y, X)
    Raises
    ------
    ValueError
        if the ome xml in the file has an invaid dimension-order.
    """
    with tifffile.TiffFile(filename) as tif:
        array = tif.asarray()
        ome = ome_types.to_dict(tif.ome_metadata, parser="lxml")
        ome = ome_types.OME(**ome)

    # ome = ome_types.from_tiff(filename)
    pixels = ome.images[0].pixels

    t, c, z, y, x = (
        pixels.size_t,
        pixels.size_c,
        pixels.size_z,
        pixels.size_y,
        pixels.size_x,
    )

    assert array.ndim >= 2 and array.shape[-2:] == (y, x)

    # makes the array in (T, C, Z, Y, X) order
    if pixels.dimension_order == DimensionOrder.XYZCT:
        array = array.reshape(t, c, z, y, x)
    elif pixels.dimension_order == DimensionOrder.XYCZT:
        array = array.reshape(t, z, c, y, x)
        array = np.moveaxis(array, 1, 2)
    elif pixels.dimension_order == DimensionOrder.XYCTZ:
        array = array.reshape(z, t, c, y, x)
        array = np.moveaxis(array, 0, 2)
    elif pixels.dimension_order == DimensionOrder.XYZTC:
        array = array.reshape(c, t, z, y, x)
        array = np.moveaxis(array, 0, 1)
    elif pixels.dimension_order == DimensionOrder.XYTZC:
        array = array.reshape(c, z, t, y, x)
        array = np.moveaxis(array, 2, 0)
    elif pixels.dimension_order == DimensionOrder.XYTCZ:
        array = array.reshape(z, c, t, y, x)
        array = np.transpose(array, (2, 1, 0, 3, 4))
    else:
        raise ValueError(f"Unknown dimension order: {pixels.dimension_order}")

    pixels.dimension_order = DimensionOrder.XYZCT

    return array, ome


def write_ome_tiff(
    filename: Union[str, os.PathLike, BinaryIO, tifffile.FileHandle],
    array: np.ndarray,
    ome: Optional[ome_types.OME] = None,
    compress_zlib: bool = True,
) -> None:
    """_summary_

    Parameters
    ----------
    filename : Union[str, os.PathLike, BinaryIO, tifffile.FileHandle]
        File name or writable binary stream, such as an open file or BytesIO.
    array : np.ndarray
        a 5D numpy array which is in the format: (T, C, Z, Y, X)
    ome : Optional[ome_types.OME], optional
        an OME metadata, by default None
    compress_zlib : bool, optional
        whether to compress pixel data by using the ZLIB algorithm, by default True
    """
    assert array.ndim == 5, (
        "array dimension should be 5 and "
        f"its shape is (T, C, Z, Y, X), but {array.ndim}"
    )

    if ome is None:
        ome = _gen_ome_metadata(array)

    tifffile.imwrite(
        filename,
        array,
        photometric="minisblack",
        description=ome.to_xml().encode(),
        bigtiff=True,
        metadata=None,
        compression="zlib" if compress_zlib else None,
    )


def export_as_ome_tiff(
    image: ImageWrapper,
    filename: Union[str, os.PathLike, BinaryIO],
    compress_zlib: bool = True,
) -> None:
    ome = export_ome_metadata(image)
    array5d = _get_ndarray(image)
    plane_count = array5d.shape[0] * array5d.shape[1] * array5d.shape[2]

    pixels = ome.images[0].pixels
    pixels.dimension_order = ome_model.pixels.DimensionOrder.XYZCT
    pixels.metadata_only = False
    pixels.tiff_data_blocks.append(ome_model.TiffData(plane_count=plane_count))

    write_ome_tiff(filename, array5d, ome=ome, compress_zlib=compress_zlib)


def export_ome_metadata(image: ImageWrapper) -> ome_types.OME:
    assert image.getId() is not None, "no image ID"
    assert image.getPrimaryPixels().getId() is not None, "no Pixels ID"
    assert image._conn is not None

    ome = ome_types.OME()
    export_image_metadata(image_obj=image, conn=image._conn, ome=ome, in_place=True)

    return ome


def _get_ndarray(image: ImageWrapper) -> np.ndarray:
    image_shape = (
        image.getSizeT(),
        image.getSizeC(),
        image.getSizeZ(),
        image.getSizeY(),
        image.getSizeX(),
    )
    type_str = image.getPrimaryPixels().getPixelsType().getValue()
    out_array = np.empty(image_shape, dtype=pixel_type_to_numpy(type_str))

    zct_list = []
    for z in range(image.getSizeZ()):
        for c in range(image.getSizeC()):
            for t in range(image.getSizeT()):
                zct_list.append((z, c, t))

    for i, plane in enumerate(image.getPrimaryPixels().getPlanes(zct_list)):
        z, c, t = zct_list[i]
        out_array[t, c, z] = plane

    return out_array
