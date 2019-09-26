#!/bin/env python

from astropy.io import fits
import astropy.units as u
import numpy
import desimodel.focalplane.geometry as fgeom
from sklearn.neighbors import KDTree
import sys

class fiber_selector:

    def __init__(self, filename):
        self.geom_tbl_  = fgeom.load_deviceloc()
        self.x_tbl_     = self.geom_tbl_['X']
        self.y_tbl_     = self.geom_tbl_['Y']
        self.petal_tbl_ = self.geom_tbl_['PETAL']
        self.geom_tree_ = KDTree(list(zip(self.x_tbl_, self.y_tbl_)), leaf_size=4)
        self.filename_  = filename
        self.LUT_       = numpy.load(self.filename_)
        self.x_pos_     = self.LUT_['x_pos']
        self.y_pos_     = self.LUT_['y_pos']
        devices, distances = self.geom_tree_.query_radius(numpy.asarray(list(zip(self.x_pos_, self.y_pos_))).reshape(len(self.x_pos_), 2), r=12,
                                                          sort_results=True, return_distance=True)
        self.petals_ = numpy.zeros(len(self.x_pos_))
        for i in range(len(self.x_pos_)):
            self.petals_[i] = self.petal_tbl_[devices[i][0]]

    def report(self):
        for i in range(10):
            print("Number of fibers on petal {} is : {}".format(i, len(self.petals_[self.petals_==i])))
            
    def get_fibers_on_petal(self, petalid, verbose=False):
        fibers = numpy.where(self.petals_==petalid)[0]
        if verbose:
            print(fibers)
        return fibers

    def get_random_fibers_on_petal(self, petalid, num_fibers, verbose=False):
        all_fibers = self.get_fibers_on_petal(petalid, verbose)
        selected_fibers = numpy.random.choice(all_fibers, num_fibers)
        unique_size = len(numpy.unique(selected_fibers))
        return numpy.asarray(selected_fibers, dtype=int)

    def get_random_fibers(self, num_fiber_per_petal, verbose=False):
        selected_fibers = numpy.zeros(num_fiber_per_petal * 10, dtype=int)
        for i in range(10):
            curr_fibers = self.get_random_fibers_on_petal(i, num_fiber_per_petal,verbose)
            selected_fibers[i*num_fiber_per_petal:(i+1)*num_fiber_per_petal] = curr_fibers
        return selected_fibers
            
if __name__ == "__main__":
    filename = sys.argv[1]
    selector = fiber_selector(filename)
    print(selector.get_fibers_on_petal(0))
    print(selector.get_random_fibers_on_petal(0, 10))
    print(selector.get_random_fibers(10))
