from osgeo import gdal
import os
import subprocess

input_tif = "download-4/6a3bfdd89e3416fefefdf90b.tif"
output_cog = "download-4/6a3bfdd89e3416fefefdf90b_cog_zstd_256.tif"


if os.path.exists(output_cog):
    os.remove(output_cog)

gdal.Translate(
    destName=output_cog,
    srcDS=input_tif,
    format="COG",
    creationOptions=[
        "COMPRESS=ZSTD",
        "LEVEL=22",
        "BLOCKSIZE=512",
        "PREDICTOR=3",
        "OVERVIEWS=AUTO",
        "OVERVIEW_RESAMPLING=AVERAGE"
    ],
)

print(f"Done: {output_cog}")
