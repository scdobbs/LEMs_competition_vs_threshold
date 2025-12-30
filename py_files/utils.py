import copy
import numpy as np

def crop(dem, area, flow_direction, extent_indexes = (200, 800, 200, 800)):

    rowscols_minmin = [extent_indexes[0], extent_indexes[2]]
    rowscols_maxmax = [extent_indexes[1], extent_indexes[3]]

    extent_minmin = dem._rowscols_to_xy((rowscols_minmin,))[0]
    extent_maxmax = dem._rowscols_to_xy((rowscols_maxmax,))[0]

    extent = (extent_minmin[0], extent_maxmax[0], extent_maxmax[1], extent_minmin[1])

    dem_clip = dem.clip_to_extent(extent)
    area_clip = area.clip_to_extent(extent)
    fd_clip = flow_direction.clip_to_extent(extent)

    return dem_clip, area_clip, fd_clip

