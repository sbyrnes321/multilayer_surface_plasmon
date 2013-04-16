# -*- coding: utf-8 -*-
"""
Calculates surface-plasmon-polariton modes in multilayer planar structures.

For more details see: http://pythonhosted.org/multilayer_surface_plasmon/
"""
#Copyright (C) 2013 Steven Byrnes
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division, print_function
import numpy as np
import math, cmath
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt
inf = float('inf')
from math import pi
from copy import deepcopy

import numericalunits as nu
# numericalunits is a package for units and constants,
# See https://pypi.python.org/pypi/numericalunits
# How it works in three sentences:
# (1) 4 * nu.cm means "4 cm".
# (2) my_length / nu.um means "my_length expressed in microns"
# (3) If any output randomly varies between python sessions, it means you made
# a dimensional-analysis error.

def floats_are_equal(a, b, tol=1e-5):
    """
    Checks whether the floats are equal, to within tol relative error. If so,
    return true. If not, print both floats and return false. This function also
    accepts complex inputs. Expected use: "assert floats_are_equal(x,y)"
    """
    if abs(a - b) <= tol * (abs(a) + abs(b)):
        return True
    else:
        print(a, b)
        return False

def find_all_zeros(min_re, max_re, min_im, max_im, fn,
                   grid_points, iterations, reduction_factor,
                   plot_full_region, show_progress):
    """
    fn is a complex function of a complex parameter fn(z). This function tries
    to find all its zeros. Looks in the search space given by
    min_re <= Re(z) <= max_re and min_im <= Im(z) <= max_im. (But it may also
    return some minima slightly outside that search space.)
    
    show_progress=True prints algorithm status messages.
    
    plot_full_region=True displays two color diagrams of the full region
    [min_re, max_re] x [min_im, max_im]. The first is a log-plot of |f(z)|. The
    second uses complex analysis to plot a quantity that often makes the zeros
    of f(z) stand out a little better. (The details don't matter, it's just a
    nice visual.)
        
    The algorithm is very simple: We're looking in a rectangular
    region in the complex plane. We evaluate fn(z) at a grid of 20 x 20 points
    within that region (replace "20" with "grid_points"). Each point with
    |fn(z)| smaller than its eight neighbors is a candidate local minimum,
    so we draw a smaller box around it, reduced in each dimension by
    reduction_factor. Repeat this process a number of times given by the
    iterations parameter. (In each step, the number of boxes under
    investigation can increase or decrease based on how many candidate minima
    were discovered in the previous step.)
    
    The final accuracy in Re(z) is something like
    (max_re - min_re) / (grid_points * reduction_factor**(iterations-1))
    Analogously for Im(z).
    
    Returns a list of complex answers: [z0, z1, ...]. Some may be spurious, so
    check each before use.
        
    The code itself is totally generic, but graph captions etc assume that the
    fn(z) is really fn(kx), the complex in-plane wavenumber, and it uses
    units of radians per micron
    """
    # Check arguments
    assert reduction_factor > 1 and max_re > min_re and max_im > min_im
    assert (max_re.imag == 0 and min_re.imag == 0
            and max_im.imag == 0 and min_im.imag == 0)
    # Edge-point rejection (see below) relies on the following assumption:
    assert grid_points > 2 * reduction_factor
    
    
    if plot_full_region:
    
        def inverse_fn(z):
            """ 1 / fn(z) """
            f = fn(z)
            return inf if f == 0 else 1/f
            
        def contour_int(z, d_re, d_im):
            """
            Approximate the contour integral of inv_fn around a point z, using
            a rectangle of half-width d_re (in real direction) and half-height
            d_im. This makes 
            """
            assert d_re.imag == 0 and d_im.imag == 0 and d_re > 0 and d_im > 0
            below = inverse_fn(z - 1j * d_im)
            above = inverse_fn(z + 1j * d_im)
            left = inverse_fn(z - d_re)
            right = inverse_fn(z + d_re)
            return (below * (2 * d_re) + right * (2j * d_im)
                    + above * (-2 * d_re) + left * (-2j * d_im))
        
        res, re_step = np.linspace(min_re, max_re, num=100, retstep=True)
        ims, im_step = np.linspace(min_im, max_im, num=100, retstep=True)
        
        fig = plt.figure()
        direct_plot = fig.add_subplot(111)
        data = [[math.log10(abs(fn(re + 1j * im))) for re in res] for im in ims]
        direct_plot.imshow(data, extent=(min_re * nu.um, max_re * nu.um,
                                         min_im * nu.um, max_im * nu.um),
                           origin='lower')
        direct_plot.set_xlabel('Re(kx) [rad/um]')
        direct_plot.set_ylabel('Im(kx) [rad/um]')
        direct_plot.set_title('log(|fn(z)|) -- Looking for minima (blue)')

        fig = plt.figure()
        contour_plot = fig.add_subplot(111)
        data = [[-math.log10(abs(contour_int(re + 1j * im, re_step, im_step)))
                                                  for re in res] for im in ims]
        contour_plot.imshow(data, extent=(min_re * nu.um, max_re * nu.um,
                                         min_im * nu.um, max_im * nu.um),
                           origin='lower')
        contour_plot.set_xlabel('Re(kx) [rad/um]')
        contour_plot.set_ylabel('Im(kx) [rad/um]')
        contour_plot.set_title(
             '-log(|contour integral of 1/fn(z) around a little rectangle|)\n'
           + ' -- This plot highlights zeros in fn(z), but also lines of\n'
           + 'discontinuity (where top or bottom kz is pure-imaginary)')
    
    # "regions" is a list where each entry has the form
    # [min_re, max_re, min_im, max_im]. Each entry describes a region in which we
    # are seeking local minima.
    regions = [[min_re, max_re, min_im, max_im]]
    
    region_width_re = max_re - min_re
    region_width_im = max_im - min_im
    
    for iteration_number in range(iterations):
        # all_local_mins will be a list of (x, y) for every local minimum in
        # every region. This is used to generate the next iteration.
        all_local_mins = []
        for region_index in range(len(regions)):
            min_re_now, max_re_now, min_im_now, max_im_now = regions[region_index]
            results_grid = []
            re_list, re_step = np.linspace(min_re_now, max_re_now, num=grid_points, retstep=True)
            im_list, im_step = np.linspace(min_im_now, max_im_now, num=grid_points, retstep=True)
            fn_to_minimize = lambda z : abs(fn(z))
            
            results_grid = [[fn_to_minimize(re + 1j * im) for im in im_list]
                                                             for re in re_list]
            results_grid = np.array(results_grid)
            # local_mins will be a list of (i,j) where (re_list[i], im_list[j])
            # is a local minimum on the results_grid
            local_mins = []
            for i in range(grid_points):
                for j in range(grid_points):
                    is_min = all(results_grid[i2, j2] >= results_grid[i,j]
                                    for i2 in [i-1, i, i+1]
                                      for j2 in [j-1, j, j+1]
                                        if (0 <= i2 < grid_points
                                             and 0 <= j2 < grid_points))
                    if is_min:
                        local_mins.append((i,j))
            # local_mins_OK is the subset of local_mins that passes the
            # the edge-rejection test.
            # The edge-rejection test says that after the 0'th iteration, any
            # point at an edge is probably not a true minimum.
            
            local_mins_OK = []
            for (i,j) in local_mins:
                z_now = re_list[i] + 1j * im_list[j]
                if iteration_number >= 2 and (i == 0 or j == 0 or
                                    i == grid_points-1 or j == grid_points-1):
                    # Rejecting an edge point...
                    if show_progress:
                        print('----')
                        print('Deleting edge point: region #'
                              + str(region_index+1) + '  (i,j)=', (i,j),
                              '  kx in rad/um=',
                              z_now / nu.um**-1,
                              '  fn(z)=', fn(z_now))
                else:
                    local_mins_OK.append((i,j))
            
            # Add local_mins_OK entries into all_local_mins
            for (i,j) in local_mins_OK:
                all_local_mins.append(re_list[i] + 1j * im_list[j])
            
            if show_progress:
                print('----')
                print('iter #' + str(iteration_number)
                    + ' , region #' + str(region_index+1) + ' of ' + str(len(regions))
                    + ' , ' + str(len(local_mins_OK)) + ' minima')
                if len(local_mins_OK) > 0:
                    print('For each, here is ((i, j), kx in rad/um, fn(kx)):')
                    print([((i, j), (re_list[i] + 1j * im_list[j]) / nu.um**-1,
                                                  fn(re_list[i] + 1j * im_list[j]))
                                                      for (i,j) in local_mins_OK])

        # Now we've gone through every region.
        # Delete redundant minima that showed up in overlapping regions.
        all_local_mins_norepeat = []
        def is_repeat(z1, z2):
            return ((abs((z1 - z2).real) <= 0.5 * re_step) and
                    (abs((z1 - z2).imag) <= 0.5 * im_step))
        for z_now in all_local_mins:
            if not any(is_repeat(z_now, z) for z in all_local_mins_norepeat):
                all_local_mins_norepeat.append(z_now)
        if show_progress:
            num_deleted = len(all_local_mins) - len(all_local_mins_norepeat)
            if num_deleted > 0:
                print('----')
                print('After iter #' + str(iteration_number)
                    + ', deleted ' + str(num_deleted) + ' redundant point(s)')

        all_local_mins = all_local_mins_norepeat
        
        if show_progress:
            print('----')
            print('** After iter #' + str(iteration_number) + ', we have '
                  + str(len(all_local_mins)) + ' candidate minima')
        
        region_width_re /= reduction_factor
        region_width_im /= reduction_factor
        
        regions = [[z.real - region_width_re / 2, z.real + region_width_re / 2,
                    z.imag - region_width_im / 2, z.imag + region_width_im / 2]
                        for z in all_local_mins]
    
    # Done with main algorithm. Show the discovered minima on the plots as
    # white X's. Note: Zeros outside the plot region will not be seen here,
    # but the function still returns them.
    if plot_full_region:
        # Keep the image filling the plot area
        direct_plot.autoscale(False)
        contour_plot.autoscale(False)
        for z in all_local_mins:
            direct_plot.plot(z.real * nu.um, z.imag * nu.um, 'wx')
            contour_plot.plot(z.real * nu.um, z.imag * nu.um, 'wx')
    return all_local_mins
                    

def find_kzs(params):
    """
    "params" is a dictionary containing w (angular frequency), kx (angular
    wavenumber), ex_list (unitless permittivity of each layer in x-direction),
    ez_list (ditto in z direction), mu_list (unitless permeability in
    y-direction).
    
    This function returns a new dictionary containing all those data PLUS
    kz_list, a list of kz in each layer.
    """
    w = params['w'] # angular frequency (w looks like omega)
    kx = params['kx']
    ex_list = params['ex_list']
    ez_list = params['ez_list']
    mu_list = params['mu_list']
    N = len(ez_list)
    assert N == len(ex_list) == len(ez_list) == len(mu_list) >= 2
    assert w > 0
    for list_name in ['ex_list', 'ez_list', 'mu_list']:
        for i in range(N):
            assert params[list_name][i].imag >= 0

    kz_list = [cmath.sqrt(w**2 * ex_list[i] * mu_list[i] / nu.c0**2
                         - kx**2 * ex_list[i] / ez_list[i]) for i in range(N)]
    # Imaginary parts should be nonnegative
    kz_list = [(-kz if kz.imag < 0 else kz) for kz in kz_list]
    
    new_params = deepcopy(params)
    new_params['kz_list'] = kz_list
    return new_params
    
def bc_matrix(params):
    """
    Calculate the "boundary condition matrix". This is a matrix M such that
    
    M * [[H0down],[H1up],[H1down],...] = [[0],[0],...]
    
    IF the boundary conditions are all satisfied. (See online docs for
    definitions and what's going on.)
    
    params should contain ex_list, ez_list, kx, kz_list, d_list (thickness of
    each layer, first and last should be inf.)
    """
    w = params['w']
    kx = params['kx']
    d_list = params['d_list']
    ex_list = params['ex_list']
    ez_list = params['ez_list']
    kz_list = params['kz_list']
    N = len(d_list)
    assert N == len(d_list) == len(ex_list) == len(ez_list) == len(kz_list)
    assert N >= 2
    assert d_list[0] == d_list[-1] == inf
    
    # delta = e^{i * kz * d}, i.e. phase change across each layer
    # delta[0] and delta[-1] are undefined and are not used.
    delta_list = [cmath.exp(1j * kz_list[i] * d_list[i]) for i in range(N)]
    
    Ex_up_over_H_up_list = [kz_list[i] / (w * ex_list[i] * nu.eps0)
                                                           for i in range(N)]
    Ex_down_over_H_down_list = [-a for a in Ex_up_over_H_up_list]
    Ez_up_over_H_up_list = [-kx / (w * ez_list[i] * nu.eps0) for i in range(N)]
    Ez_down_over_H_down_list = Ez_up_over_H_up_list[:]
    
    mat = np.zeros((2*N-2, 2*N-2), dtype=complex)
    
    for row_now in range(N-1):
        # This row concerns continuity of Ex across the boundary between
        # layer_under and layer_over (under and over the boundary respectively)
        layer_under = row_now
        layer_over = layer_under + 1
        # up_under_index is the column index in mat that gets multiplied by
        # H_{up} in layer_under.
        up_under_index = 2 * layer_under - 1
        down_under_index = 2 * layer_under
        up_over_index = 2 * layer_over - 1
        down_over_index = 2 * layer_over
        
        if layer_under != 0:
            assert 0 <= up_under_index < 2*N-2
            mat[row_now, up_under_index] = (
                  Ex_up_over_H_up_list[layer_under] * delta_list[layer_under])
        mat[row_now, down_under_index] = Ex_down_over_H_down_list[layer_under]
        mat[row_now, up_over_index] = -Ex_up_over_H_up_list[layer_over]
        if layer_over != N-1:
            assert 0 <= down_over_index < 2*N-2
            mat[row_now, down_over_index] = (
                -Ex_down_over_H_down_list[layer_over] * delta_list[layer_over])

    for row_now in range(N-1, 2*N-2):
        # This row concerns continuity of eps_z * Ez across the boundary between
        # layer_under and layer_over (under and over the boundary respectively)
        layer_under = row_now - (N-1)
        layer_over = layer_under + 1
        # up_under_index is the column index in mat that gets multiplied by
        # H_{up} in layer_under.
        up_under_index = 2 * layer_under - 1
        down_under_index = 2 * layer_under
        up_over_index = 2 * layer_over - 1
        down_over_index = 2 * layer_over
        
        if layer_under != 0:
            assert 0 <= up_under_index < 2*N-2
            mat[row_now, up_under_index] = (ez_list[layer_under] *
                   Ez_up_over_H_up_list[layer_under] * delta_list[layer_under])
        mat[row_now, down_under_index] = (ez_list[layer_under] *
                                         Ez_down_over_H_down_list[layer_under])
        mat[row_now, up_over_index] = (-ez_list[layer_over] * 
                                              Ez_up_over_H_up_list[layer_over])
        if layer_over != N-1:
            assert 0 <= down_over_index < 2*N-2
            mat[row_now, down_over_index] = (-ez_list[layer_over] *
                 Ez_down_over_H_down_list[layer_over] * delta_list[layer_over])
    
    return mat

def find_kx(input_params, search_domain=None, show_progress=False,
            grid_points=20, iterations=8, reduction_factor=5,
            plot_full_region=True):
    """
    input_params is a dictionary with the simulation parameters. (ex_list,
    d_list, etc.) Returns a list of possible complex kx, sorted from the
    lowest-order mode to the highest one discovered.
    
    search_domain is [min Re(kx), max Re(kx), min Im(kx), max Im(kx)] in which
    to search for solutions. With default (None), I use some heuristics to
    guess a region that is likely to find at least the first mode or two.
    
    The following parameters are passed straight into find_all_zeros():
    show_progress, grid_points, iterations, reduction_factor, and
    plot_full_region. show_progress=True prints diagnostics during search for kx minima.
    """
    w = input_params['w']
    d_list = input_params['d_list']
    ex_list = input_params['ex_list']
    ez_list = input_params['ez_list']
    mu_list = input_params['mu_list']
    N = len(mu_list)
    assert N == len(d_list) == len(ex_list) == len(ez_list)
    # error(z) approaches 0 as kx = z approaches a true plasmon mode.
    # It's proportional to the determinant of the boundary-condition matrix, 
    # which equals zero at modes.
    def error(kx):
        if kx == 0:
            return inf
        temp_params = input_params.copy()
        temp_params['kx'] = kx
        should_be_zero = np.linalg.det(bc_matrix(find_kzs(temp_params)))
        return should_be_zero / kx**(N+1)
        # "return should_be_zero" is also OK but has an overall slope that
        # makes it harder to find zeros; also, there's a false-positive at k=0.
    
    # choose the region in which to search for minima. My heuristic is:
    # The upper limit of kx should be large enough that
    # 2 * pi * i * kzm * d ~ 20 for the thinnest layer we have, or 3 times
    # the light-line, whichever is bigger.
    if search_domain is None:
        kx_re_max = max(max(abs((20 / (2 * pi * d_list[i]))
                        * cmath.sqrt(ez_list[i] / ex_list[i])) for i in range(1,N)),
                    3 * w / nu.c0)
        kx_re_min = -kx_re_max
        kx_im_min = 0
        kx_im_max = abs(kx_re_max)
    else:
        kx_re_min = search_domain[0]
        kx_re_max = search_domain[1]
        kx_im_min = search_domain[2]
        kx_im_max = search_domain[3]
    
    # Main part of function: Call find_all_zeros()
    kx_list = find_all_zeros(kx_re_min, kx_re_max, kx_im_min, kx_im_max, error,
                           show_progress=show_progress, grid_points=grid_points,
                           iterations=iterations,
                           reduction_factor=reduction_factor,
                           plot_full_region=plot_full_region)
    
    # sort and remove "repeats" with opposite signs
    kx_list = sorted(kx_list, key=(lambda kx : abs(kx)))
    i=0
    while i < len(kx_list) - 1:
        if abs(kx_list[i] + kx_list[i+1]) <= 1e-6 * (abs(kx_list[i]) + abs(kx_list[i+1])):
            kx_list.pop(i)
        else:
            i += 1
    
    # Fix amplifying waves
    kx_list = [(-kx if (kx.imag < 0 or (kx.imag==0 and kx.real < 0)) else kx)
                                                            for kx in kx_list]
    
    return kx_list

def find_all_params_from_kx(params):
    """
    params is a dictionary containing kx and other simulation parameters like
    w, d_list, etc. It is assumed that this kx really is a mode!
    
    This function calculates kz_list, H_up_list, H_down_list, Ex_up_list,
    Ex_down_list, Ez_up_list, Ez_down_list.
    It returns a new parameter dictionary containing all the old information
    plus those newly-calculated parameters.
    
    This is linear optics, so you can scale the E and H up or down by any
    constant factor. (And Poynting vector by the square of that factor.)
    I chose the normalization that makes the maximum of Ez_up_list equal to
    1 V/nm. (This is arbitrary.)
    
    layer_bottom_list[i] is the z-coordinate of the bottom of layer i. Assume
    that layer 0 is z<0,
    layer 1 is 0 < z < d_list[1],
    layer 2 is d_list[1] < z < d_list[1] + d_list[2], etc.
    """
    new_params = find_kzs(deepcopy(params))
    w = new_params['w']
    d_list = new_params['d_list']
    kx = new_params['kx']
    kz_list = new_params['kz_list']
    ex_list = new_params['ex_list']
    ez_list = new_params['ez_list']
    mu_list = new_params['mu_list']
    N = len(mu_list)
    
    mat = bc_matrix(new_params)
    eigenvals, eigenvecs = np.linalg.eig(mat)
    which_eigenval_is_zero = np.argmin(np.abs(eigenvals))
    null_vector = eigenvecs[:,which_eigenval_is_zero]
    if False:
        print('null vector:')
        print(null_vector)
        print('matrix entry absolute values:')
        print(np.abs(mat))
        print('abs(mat . null_vector) should be 0:')
        print(np.abs(np.dot(mat, null_vector)))
        print('calculated eigenvalue:')
        print(eigenvals[which_eigenval_is_zero])
    H_up_list = [0]
    H_up_list.extend(null_vector[i] for i in range(1, 2*N-2, 2))
    H_down_list = [null_vector[i] for i in range(0, 2*N-2, 2)]
    H_down_list.append(0)
    assert N == len(H_up_list) == len(H_down_list)
    
    Ex_up_list = [H_up_list[i] * kz_list[i] / (w * ex_list[i] * nu.eps0)
                                                            for i in range(N)]
    Ex_down_list = [-H_down_list[i] * kz_list[i] / (w * ex_list[i] * nu.eps0)
                                                            for i in range(N)]
    Ez_up_list = [-H_up_list[i] * kx / (w * ez_list[i] * nu.eps0)
                                                            for i in range(N)]
    Ez_down_list = [-H_down_list[i] * kx / (w * ez_list[i] * nu.eps0)
                                                            for i in range(N)]
    
    # normalize E and H.
    largest_Ez_up_index = np.argmax(np.abs(np.array(Ez_up_list)))
    scale_factor = (1 * nu.V/nu.nm) / Ez_up_list[largest_Ez_up_index]
    for X_list in [H_up_list, H_down_list, Ex_up_list, Ex_down_list,
                   Ez_up_list, Ez_down_list]:
        for i in range(N):
            X_list[i] *= scale_factor
    new_params['H_up_list'] = H_up_list
    new_params['H_down_list'] = H_down_list
    new_params['Ex_up_list'] = Ex_up_list
    new_params['Ex_down_list'] = Ex_down_list
    new_params['Ez_up_list'] = Ez_up_list
    new_params['Ez_down_list'] = Ez_down_list
    
    # x-component of complex Poynting vector, integrated over a layer
    Sx_list = []
    for i in range(N):
        Ez_up = Ez_up_list[i]
        Ez_down = Ez_down_list[i]
        H_up_star = H_up_list[i].conjugate()
        H_down_star = H_down_list[i].conjugate()
        kz = kz_list[i]
        d = d_list[i]
        Sx = 0
        # add each term only if it's nonzero, to avoid 0 * nan in top and
        # bottom layers
        if Ez_up * H_up_star != 0:
            Sx += ((-Ez_up * H_up_star) / (4 * kz.imag)
                    * (1 - cmath.exp(-2 * kz.imag * d)))
        if Ez_down * H_down_star != 0:
            Sx += ((-Ez_down * H_down_star) / (4 * kz.imag)
                    * (1 - cmath.exp(-2 * kz.imag * d)))
        if Ez_down * H_up_star != 0:
            Sx += ((-Ez_down * H_up_star) / (4j * kz.real)
                   * (1 - cmath.exp(-2j * kz.real * d))
                   * cmath.exp(1j * kz * d))
        if Ez_up * H_down_star != 0:
            Sx += ((-Ez_up * H_down_star) / (4j * kz.real)
                   * (1 - cmath.exp(-2j * kz.real * d))
                   * cmath.exp(1j * kz * d))
        Sx_list.append(Sx)
    new_params['Sx_list'] = Sx_list
    # x-component of complex Poynting vector, integrated over all layers
    Sx_total = sum(Sx_list)
    new_params['Sx_total'] = Sx_total
    
    layer_bottom_list = [-inf, 0]
    for i in range(1,N-1):
        layer_bottom_list.append(layer_bottom_list[-1] + d_list[i])
    
    new_params['layer_bottom_list'] = layer_bottom_list
    return new_params

def find_layer(z, params):
    """
    Return the layer index (0 through N-1) in which you find the z-coordinate
    z. At a layer boundary, returns either one of those two layers arbitrarily.
    """
    N = len(params['d_list'])
    for i in range(N):
        if z <= params['layer_bottom_list'][i]:
            return i-1
    return N-1

def Hy(z, params, x=0, layer=None):
    """
    Complex H-field at (x,z). Optional "layer" parameter forces the use of the
    formulas for fields in a certain layer, regardless of whether z is actually
    in that layer or not.
    """
    N = len(params['d_list'])
    if layer is None:
        layer = find_layer(z, params)
    H_up = params['H_up_list'][layer]
    H_down = params['H_down_list'][layer]
    kz = params['kz_list'][layer]
    kx = params['kx']
    layer_bottom = params['layer_bottom_list'][layer]
    layer_top = inf if layer == N-1 else params['layer_bottom_list'][layer + 1]
    
    if H_up == 0:
        # This is to avoid 0 * nan for infinitely-thick top or bottom layers
        up_term = 0
    else:
        up_term = H_up * cmath.exp(1j * kz * (z - layer_bottom) + 1j * kx * x)
    if H_down == 0:
        down_term = 0
    else:
        down_term = H_down * cmath.exp(1j * kz * (layer_top - z) + 1j * kx * x)
    return up_term + down_term

def Ex(z, params, x=0, layer=None):
    """
    Complex E-field (x-component) at (x,z). See Hy documentation.
    """
    N = len(params['d_list'])
    if layer is None:
        layer = find_layer(z, params)
    Ex_up = params['Ex_up_list'][layer]
    Ex_down = params['Ex_down_list'][layer]
    kz = params['kz_list'][layer]
    kx = params['kx']
    layer_bottom = params['layer_bottom_list'][layer]
    layer_top = inf if layer == N-1 else params['layer_bottom_list'][layer + 1]
    
    if Ex_up == 0:
        # This is to avoid 0 * nan for infinitely-thick top or bottom layers
        up_term = 0
    else:
        up_term = Ex_up * cmath.exp(1j * kz * (z - layer_bottom) + 1j * kx * x)
    if Ex_down == 0:
        down_term = 0
    else:
        down_term = Ex_down * cmath.exp(1j * kz * (layer_top - z) + 1j * kx * x)
    return up_term + down_term

def Ez(z, params, x=0, layer=None):
    """
    Complex E-field (z-component) at (x,z). See Hy documentation.
    """
    N = len(params['d_list'])
    if layer is None:
        layer = find_layer(z, params)
    Ez_up = params['Ez_up_list'][layer]
    Ez_down = params['Ez_down_list'][layer]
    kz = params['kz_list'][layer]
    kx = params['kx']
    layer_bottom = params['layer_bottom_list'][layer]
    layer_top = inf if layer == N-1 else params['layer_bottom_list'][layer + 1]
    
    if Ez_up == 0:
        # This is to avoid 0 * nan for infinitely-thick top or bottom layers
        up_term = 0
    else:
        up_term = Ez_up * cmath.exp(1j * kz * (z - layer_bottom) + 1j * kx * x)
    if Ez_down == 0:
        down_term = 0
    else:
        down_term = Ez_down * cmath.exp(1j * kz * (layer_top - z) + 1j * kx * x)
    return up_term + down_term

def Sx(z, params, x=0, layer=None):
    """
    Complex Poynting vector (x-component) at (x,z). See Hy documentation. The
    real part of this equals net energy flow (averaged over a cycle). The
    imaginary part has something to do with reactive stored energy at a given
    point, flowing around reversably within a single cycle.
    """
    Ez_here = Ez(z, params, x=x, layer=layer)
    Hy_here = Hy(z, params, x=x, layer=layer)
    return -0.5 * Ez_here * Hy_here.conjugate()

def check_mode(params, thorough=True):
    """
    Check that mode is valid. "thorough" mode takes a bit longer, because it
    also checks that the total Poynting vector is consistent with the numerical
    integral of the local Poynting vector.
    """
    N = len(params['d_list'])
    w = params['w']
    kx = params['kx']
    kz_list = params['kz_list']
    ex_list = params['ex_list']
    ez_list = params['ez_list']
    mu_list = params['mu_list']
    layer_bottom_list = params['layer_bottom_list']
    Sx_list = params['Sx_list']
    Sx_total = params['Sx_total']
    
    # check boundary conditions for Ex, Ez, Hy
    for layer_under in range(0,N-1):
        layer_over = layer_under + 1
        z = layer_bottom_list[layer_over]
        ez_under = ez_list[layer_under]
        ez_over = ez_list[layer_over]
        assert floats_are_equal(Ex(z, params, layer=layer_under),
                                Ex(z, params, layer=layer_over))
        assert floats_are_equal(ez_under * Ez(z, params, layer=layer_under),
                                ez_over * Ez(z, params, layer=layer_over))
        assert floats_are_equal(Hy(z, params, layer=layer_under),
                                Hy(z, params, layer=layer_over))
   
    # check a few properties of each layer
    for i in range(N):
        kz = kz_list[i]
        ez = ez_list[i]
        ex = ex_list[i]
        mu = mu_list[i]
        assert floats_are_equal(kz**2,
                                w**2 * mu * ex / nu.c0**2 - kx**2 * ex /ez)
        if i == 0 or i == N-1:
            assert kz.imag > 0
        else:
            assert kz.imag >= 0
    
    if thorough:
        # Check Sx_list against a numerical integration.
        # Numerical integration expects order-unity integrand, or else the
        # absolute-error criterion can fire before convergence. (A few orders
        # of magnitude away from 1 is OK, but not 20 orders of magnitude.) So
        # I'll scale up before integrating, then scale down by the same factor
        # afterwards. Poor integration can flag a correct solution as incorrect,
        # but not vice-versa: If it passes the test, you can trust it.
        
        # This scale_factor seems to work pretty reliably
        scale_factor = max(abs(Sx(0, params, layer=0)),
                           abs(Sx(0, params, layer=1)))
        assert scale_factor != 0
        
        for i in range(N):
            # Calculate integration limits
            if i != 0:
                lower_z = layer_bottom_list[i]
            else:
                lower_z = -20 / abs(kz_list[i].imag)
            if i != N-1:
                upper_z = layer_bottom_list[i+1]
            else:
                upper_z = 20 / abs(kz_list[i].imag)
            
            integrand_re = lambda z : (Sx(z, params) / scale_factor).real
            integrand_im = lambda z : (Sx(z, params) / scale_factor).imag
            Sx_integrated = (scipy.integrate.quad(integrand_re, lower_z, upper_z)[0]
                      + 1j * scipy.integrate.quad(integrand_im, lower_z, upper_z)[0])
            Sx_integrated *= scale_factor
            assert floats_are_equal(Sx_list[i], Sx_integrated)
    assert floats_are_equal(Sx_total, sum(Sx_list))

def plot_mode(params, filename_x=None, filename_z=None):
    """
    params is a dictionary that should include kx, w, kz_list, H_up_list, etc.
    This function plots the mode. Pass a filename_x to save the plot of Ex,
    and/or filename_z to save the plot of Ez
    """
    kz_list = params['kz_list']
    layer_bottom_list = params['layer_bottom_list']
    N = len(kz_list)
    # Choose a range of z to plot:
    if N == 2:
        # For 2 layers, put the boundary in the middle, and show the wave
        # decaying on both sides
        z_max = min(4 / abs(kz.imag) for kz in kz_list)
        z_min = -z_max
    else:
        # For >= 3 layers, the layers should take up central half of plot
        z_max = 1.5 * layer_bottom_list[-1]
        z_min = -0.5 * layer_bottom_list[-1]
    # Calculate the data
    zs = np.linspace(z_min, z_max, num=200)
    Exs = np.array([Ex(z, params) for z in zs])
    Ezs = np.array([Ez(z, params) for z in zs])
    # Normalize the E-fields to max 1
    max_E = max(abs(Exs).max(), abs(Ezs).max())
    Exs = Exs / max_E
    Ezs = Ezs / max_E
    
    plt.figure()
    plt.plot(zs / nu.nm, Exs.real, zs / nu.nm, Exs.imag)
    for i in range(1,N):
        plt.axvline(x=layer_bottom_list[i] / nu.nm, color='k')
    plt.title('Ex profile at time 0 and 1/4 cycle later')
    plt.xlabel('Position (nm)')
    plt.ylabel('E-field (arbitrary units)')
    if filename_x is not None:
        plt.savefig(filename_x)
    
    plt.figure()
    plt.plot(zs / nu.nm, Ezs.real, zs / nu.nm, Ezs.imag)
    for i in range(1,N):
        plt.axvline(x=layer_bottom_list[i] / nu.nm, color='k')
    plt.title('Ez profile at time 0 and 1/4 cycle later')
    plt.xlabel('Position (nm)')
    plt.ylabel('E-field (arbitrary units)')
    if filename_z is not None:
        plt.savefig(filename_z)

def rescale_fields(factor, params):
    """
    params is a dictionary that should include kx, w, kz_list, H_up_list, etc.
    This function multiplies the amplitude of the wave by "factor", and returns
    a new, updated parameter bundle.
    """
    new_params = deepcopy(params)
    N = len(new_params['d_list'])
    for name in ['H_up_list', 'H_down_list', 'Ex_up_list', 'Ex_down_list',
                 'Ez_up_list', 'Ez_down_list']:
        for i in range(N):
            new_params[name][i] *= factor
    for i in range(N):
        new_params['Sx_list'][i] *= abs(factor)**2
    new_params['Sx_total'] *= abs(factor)**2
    return new_params

#########################################################################
############################# TESTS #####################################
#########################################################################

def test_2_layer():
    """
    test this calculation against analytical expressions when N=2 for an
    isotropic, non-magnetic medium
    """
    # angular frequency in radians * THz
    w = 100 * nu.THz
    # Relative permittivity of metal and dielectric
    em = -4.56 + 0.12j
    ed = 1.23 + 0.01j
    ex_list = ez_list = [ed, em]
    # Relative permeabilities
    mu_list = [1,1]
    # Dictionary of input parameters
    input_params = {'w': w, 'd_list': [inf,inf], 'ex_list': ex_list,
              'ez_list': ez_list, 'mu_list': mu_list}
    
    # Calculate the theoretical kx
    theo_kx = (w / nu.c0) * cmath.sqrt((em * ed) / (em + ed))
    if theo_kx.imag < 0:
        theo_kx *= -1
    print('Theoretical kx:',
          '(%.7g+%.7gj) rad/um' % (theo_kx.real / nu.um**-1, theo_kx.imag / nu.um**-1))
    
    # If I use the theoretical kx value, the mode should be correct and
    # all my tests should pass.
    params = deepcopy(input_params)
    params['kx'] = theo_kx
    params = find_all_params_from_kx(params)
    kzd, kzm = params['kz_list']
    # check that kz_list is correct
    assert floats_are_equal(kzd**2, (w**2 / nu.c0**2) * ed**2 / (em + ed))
    assert floats_are_equal(kzm**2, (w**2 / nu.c0**2) * em**2 / (em + ed))
    # check that layer_bottom_list is correct
    assert params['layer_bottom_list'][0] == -inf
    assert params['layer_bottom_list'][1] == 0
    # Check that the boundary condition matrix agrees with hand-calculation
    bc_mat = bc_matrix(params)
    # ...top-left is Ex0down / H0down
    assert floats_are_equal(bc_mat[0,0], -kzd / (w * ed * nu.eps0))
    # ...top-right is -Ex1up / H1up
    assert floats_are_equal(bc_mat[0,1], -kzm / (w * em * nu.eps0))
    # ...bottom-left is eps0 * Ez0down / H0down
    assert floats_are_equal(bc_mat[1,0], ed * -theo_kx / (w * ed * nu.eps0))
    # ...bottom-right is -eps1 * Ez1up / H1up
    assert floats_are_equal(bc_mat[1,1], -em * -theo_kx / (w * em * nu.eps0))
    # Check that one of the eigenvalues is almost zero (compared to the size
    # of the matrix elements).
    eigenvalues = np.linalg.eig(bc_mat)[0]
    assert abs(eigenvalues).min() / abs(bc_mat).max() < 1e-6
    # Check that the mode passes all tests.
    check_mode(params)
    # Check that I can scale the fields and it still passes all tests.
    params_scaled = rescale_fields(1.23+4.56j, params)
    check_mode(params_scaled)
    
    # Now try my kx-finding algorithm, to see if it finds the right value.
    kx_list = find_kx(input_params)
    print('kx_list:',
          ['(%.7g+%.7gj) rad/um' % (kx.real / nu.um**-1, kx.imag / nu.um**-1)
                                                          for kx in kx_list])
    kx = kx_list[0]
    assert floats_are_equal(theo_kx, kx)
    
    plot_mode(params)
    
    print('If you see this message, all the tests succeeded!!')

def test_davis():
    """
    This should reproduce T.J. Davis, 2009, "Surface plasmon modes in
    multi-layer thin-films". http://dx.doi.org/10.1016/j.optcom.2008.09.043
    The first plot should resemble Fig. 1b, and the modes should match the ones
    found in the table. There are also graphs to reproduce Fig. 2a
    """
    w = 2 * pi * nu.c0 / (780 * nu.nm)
    eps_gold = -21.19 + 0.7361j
    eps_glass = 2.310
    eps_MgF2 = 1.891
    d_list = [inf, 75 * nu.nm, 10 * nu.nm, 55 * nu.nm, 10 * nu.nm, 75 * nu.nm, inf]
    ex_list = [eps_glass, eps_gold, eps_MgF2, eps_gold, eps_MgF2, eps_gold, eps_glass]
    ez_list = ex_list
    mu_list = [1,1,1,1,1,1,1]
    params = {'w': w,
              'd_list': d_list,
              'ex_list': ex_list,
              'ez_list': ez_list,
              'mu_list': mu_list}
    kx_list = find_kx(params, show_progress=False,
                      search_domain=[-0.05/nu.nm, 0.05/nu.nm, 0, 0.4/nu.nm],
                      grid_points=20, iterations=10, reduction_factor=9,
                      plot_full_region=True)
    print('kx_list -- ' + str(len(kx_list)) + ' entries...')
    print(['(%.5g+%.5gj) rad/um' % (kx.real / nu.um**-1, kx.imag / nu.um**-1)
                                                          for kx in kx_list])
    
    # The modes discoved by Davis (Table 1 of paper)
    davis_modes = [x * nu.nm**-1 for x in [1.2969e-2 + 2.7301e-5j,
                                           1.2971e-2 + 2.7644e-5j,
                                           3.0454e-2 + 3.7872e-4j,
                                           3.2794e-2 + 4.6749e-4j,
                                          -2.1254e-4 + 5.4538e-2j,
                                           1.2634e-3 + 5.4604e-2j]]
    for i in range(len(davis_modes)):
        davis_kx = davis_modes[i]
        print('Looking for "Mode ' + str(i+1) + '" in Davis paper -- kx =',
              '(%.5g+%.5gj) rad/um' % (davis_kx.real / nu.um**-1, davis_kx.imag / nu.um**-1))
        which_kx = np.argmin(abs(np.array(kx_list) - davis_kx))
        my_kx = kx_list[which_kx]
        print('Relative error: ',
              abs(my_kx - davis_kx) / (abs(my_kx) + abs(davis_kx)))

    print('---')
    print('Are the last two modes missing? They were for me. Re-try with a')
    print('smaller search domain (zoomed towards kx=0). (By the way, ')
    print('using a larger number for grid_points would also work here.)')
    print('---')

    kx_list2 = find_kx(params, show_progress=False,
                      search_domain=[-0.05/nu.nm, 0.05/nu.nm, 0, 0.1/nu.nm],
                      grid_points=20, iterations=10, reduction_factor=9,
                      plot_full_region=True)
    print('kx_list2 -- ' + str(len(kx_list)) + ' entries...')
    print(['(%.5g+%.5gj) rad/um' % (kx.real / nu.um**-1, kx.imag / nu.um**-1)
                                                          for kx in kx_list2])
    
    for i in range(len(davis_modes)):
        davis_kx = davis_modes[i]
        print('Looking for "Mode ' + str(i+1) + '" in Davis paper -- kx =',
              '(%.5g+%.5gj) rad/um' % (davis_kx.real / nu.um**-1, davis_kx.imag / nu.um**-1))
        which_kx = np.argmin(abs(np.array(kx_list2) - davis_kx))
        my_kx = kx_list2[which_kx]
        print('Relative error: ',
              abs(my_kx - davis_kx) / (abs(my_kx) + abs(davis_kx)))
        
        new_params = deepcopy(params)
        new_params['kx'] = my_kx
        new_params = find_all_params_from_kx(new_params)
        plt.figure()
        plt.title('"Mode ' + str(i+1) + '" in Davis paper -- Plot of Re(Hy) and Im(Hy)')
        zs = np.linspace(-300 * nu.nm, 500 * nu.nm, num=400)
        Hs = np.array([Hy(z, new_params) for z in zs])
        plt.plot(zs / nu.nm, Hs.real / max(abs(Hs)),
                 zs / nu.nm, Hs.imag / max(abs(Hs)))
        plt.xlabel('z (nm)')
        plt.ylabel('Hy (arbitrary units)')

#########################################################################
############################# EXAMPLES ##################################
#########################################################################


def example1():
    """
    A 300 THz wave is in an air-metal-insulator structure, where both the
    metal and insulator are anisotropic and magnetic. For the metal,
    epsilon_x = -5+2j, epsilon_z = -3+3j, mu=1.2. For the insulator,
    epsilon_x = 10, epsilon_z = 7, mu=1.3. The metal is 40nm thick.
    Goal: Find the few lowest-order modes; for each, display the kz values and
    the relation between Ex(0,0) and the total power flow in the mode.
    """
    params = {'w': 2 * pi * 300 * nu.THz,
              'd_list': [inf, 40 * nu.nm, inf],
              'ex_list': [1, -5 + 2j, 10],
              'ez_list': [1, -3 + 3j, 7],
              'mu_list': [1, 1.2, 1.3]}
    
    kx_list = find_kx(params, show_progress=False, grid_points=30,
                      iterations=8, reduction_factor=14,
                      plot_full_region=True)
    print('kx_list: ',
          ['(%.4g+%.4gj) rad/um' % (kx.real / nu.um**-1, kx.imag / nu.um**-1)
                                                          for kx in kx_list])
    for kx in kx_list:
        new_params = deepcopy(params)
        new_params['kx'] = kx
        print('---')
        print('With kx =', '(%.4g+%.4gj) rad/um' % (kx.real / nu.um**-1, kx.imag / nu.um**-1),
              ', checking mode...')
        new_params = find_all_params_from_kx(new_params)
        print('kz in each layer:',
              ['(%.4g+%.4gj) rad/um' % (kz.real / nu.um**-1, kz.imag / nu.um**-1)
                                                for kz in new_params['kz_list']])
        try:
            check_mode(new_params)
            print('If this message appears, the mode passes all tests!')
            plot_mode(new_params)
            scale_factor = (5 * nu.nW/nu.um) / new_params['Sx_total']
            scaled_params = rescale_fields(scale_factor, new_params)
            print('If this wave carries 5 nW/um power (i.e. 5 nW travels in +x-direction')
            print('through the surface x=0, 0<y<1um, -inf<z<inf)')
            print('then |Ex(0,0)|=',
                  abs(Ex(0, scaled_params)) / (nu.V/nu.m), 'V/m')
        except:
            print('kx =', '(%.4g+%.4gj) rad/um' % (kx.real / nu.um**-1, kx.imag / nu.um**-1),
                  'seems not to be a real mode (a local minimum '
                  + 'but not a zero in the error function)')
    

def example2():
    """
    A 300 THz wave is at an air-metal interface, where the metal is anisotropic
    and magnetic. Goal: Same as example1()
    """
    params = {'w': 2 * pi * 300 * nu.THz,
              'd_list': [inf, inf],
              'ex_list': [1, -5],
              'ez_list': [1, 1],
              'mu_list': [1.1, 1.3]}
    
    kx_list = find_kx(params, show_progress=False, grid_points=30,
                      iterations=8, reduction_factor=14,
                      plot_full_region=True)
    print('kx_list: ',
          ['(%.4g+%.4gj) rad/um' % (kx.real / nu.um**-1, kx.imag / nu.um**-1)
                                                          for kx in kx_list])
    for kx in kx_list:
        new_params = deepcopy(params)
        new_params['kx'] = kx
        print('---')
        print('With kx =', '(%.4g+%.4gj) rad/um' % (kx.real / nu.um**-1, kx.imag / nu.um**-1),
              ', checking mode...')
        new_params = find_all_params_from_kx(new_params)
        print('kz in each layer:',
              ['(%.4g+%.4gj) rad/um' % (kz.real / nu.um**-1, kz.imag / nu.um**-1)
                                                for kz in new_params['kz_list']])
        try:
            check_mode(new_params)
            print('If this message appears, the mode passes all tests!')
            plot_mode(new_params)
            scale_factor = (5 * nu.nW/nu.um) / new_params['Sx_total']
            scaled_params = rescale_fields(scale_factor, new_params)
            print('If this wave carries 5 nW/um power (i.e. 5 nW travels in +x-direction')
            print('through the surface x=0, 0<y<1um, -inf<z<inf)')
            print('then |Ex(0,0)|=',
                  abs(Ex(0, scaled_params)) / (nu.V/nu.m), 'V/m')
        except:
            print('kx =', '(%.4g+%.4gj) rad/um' % (kx.real / nu.um**-1, kx.imag / nu.um**-1),
                  'seems not to be a real mode (a local minimum '
                  + 'but not a zero in the error function)')
