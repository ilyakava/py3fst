"""Plotting code for cube shaped objects.

Hyperspectral data and 3D filters
"""

import numpy as np

import plotly.offline as py
from plotly.graph_objs import *
import scipy.misc

import pdb

def get_the_slice(x,y,z, surfacecolor,  colorscale='Hot', showscale=False):
    # Greys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland,Jet,Hot,Blackbody,Earth,Electric,Viridis,CividisGreys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland,Jet,Hot,Blackbody,Earth,Electric,Viridis,Cividis
    return Surface(x=x,# https://plot.ly/python/reference/#surface
                   y=y,
                   z=z,
                   surfacecolor=surfacecolor,
                   colorscale=colorscale,
                   showscale=showscale)   
def get_lims_colors(surfacecolor):# color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)

def str_to_filename(mystr):
    return "".join([c for c in mystr if c.isalpha() or c.isdigit() or c==' ']).rstrip()

def pyplot_slices(surfcolor_z, surfcolor_y,surfcolor_x,title=None,resample_factor=50):
    """
    Args:
        surfcolor_z: img that has x on vertical, y on horiz
        surfcolor_y: img that has x on vertical, z on horiz
        surfcolor_x: img that has y on vertical, z on horiz
    """

    surfcolor_z = scipy.misc.imresize(surfcolor_z, resample_factor*np.array(surfcolor_z.shape), interp='nearest')
    surfcolor_y= scipy.misc.imresize(surfcolor_y, resample_factor*np.array(surfcolor_y.shape), interp='nearest')
    surfcolor_x = scipy.misc.imresize(surfcolor_x, resample_factor*np.array(surfcolor_x.shape), interp='nearest')

    x=np.linspace(-1,1, surfcolor_z.shape[0])
    y=np.linspace(-1,1, surfcolor_z.shape[1])
    y,x=np.meshgrid(y,x)
    z=np.zeros(x.shape)
    slice_z=get_the_slice(x,y,z, surfcolor_z)    
    
    x=np.linspace(-1,1, surfcolor_y.shape[0])
    z=np.linspace(-1,1, surfcolor_y.shape[1])
    z,x=np.meshgrid(z,x)
    y=np.zeros(x.shape)
    slice_y=get_the_slice(x,y,z, surfcolor_y)

    y=np.linspace(-1,1, surfcolor_x.shape[0])
    z=np.linspace(-1,1, surfcolor_x.shape[1])
    z,y=np.meshgrid(z,y)
    x=np.zeros(z.shape)
    slice_x=get_the_slice(x,y,z, surfcolor_x)

    # sminz, smaxz=get_lims_colors(surfcolor_z)
    # sminy, smaxy=get_lims_colors(surfcolor_y)
    # sminx, smaxx=get_lims_colors(surfcolor_x)
    # vmin=min([sminz, sminy, sminx])
    # vmax=max([smaxz, smaxy, smaxx])

    # slice_z.update(cmin=vmin, cmax=vmax)
    # slice_y.update(cmin=vmin, cmax=vmax)
    # slice_x.update(cmin=vmin, cmax=vmax, showscale=True)
    slice_x.update(showscale=True)

    axis = dict(showbackground=True, 
            backgroundcolor="rgb(230, 230,230)",
            gridcolor="rgb(255, 255, 255)",      
            zerolinecolor="rgb(255, 255, 255)",  
            )


    title = title or 'Slices in volumetric data'
    layout = Layout(
             title=title, 
             width=700,
             height=700,
             scene=Scene(xaxis=XAxis(axis),
                         yaxis=YAxis(axis), 
                         zaxis=ZAxis(axis), 
                         aspectratio=dict(x=1,
                                          y=1, 
                                          z=1
                                         ),
                        )
            )

    fig=Figure(data=Data([slice_z,slice_y,slice_x]), layout=layout)
    py.plot(fig, filename=str_to_filename(title)+'.html', auto_open=False)

def pyplot_cube(surfcolor_z, surfcolor_y,surfcolor_x,title=None,resample_factor=50):
    """
    Args:
        surfcolor_z: img that has x on vertical, y on horiz
        surfcolor_y: img that has x on vertical, z on horiz
        surfcolor_x: img that has y on vertical, z on horiz
    """

    if resample_factor != 1:
        surfcolor_z = scipy.misc.imresize(surfcolor_z, resample_factor*np.array(surfcolor_z.shape), interp='nearest')
        surfcolor_y= scipy.misc.imresize(surfcolor_y, resample_factor*np.array(surfcolor_y.shape), interp='nearest')
        surfcolor_x = scipy.misc.imresize(surfcolor_x, resample_factor*np.array(surfcolor_x.shape), interp='nearest')

    x=np.linspace(-1,1, surfcolor_z.shape[0])
    y=np.linspace(-1,1, surfcolor_z.shape[1])
    y,x=np.meshgrid(y,x)
    z=np.ones(x.shape)
    slice_z=get_the_slice(x,y,z, surfcolor_z)    
    
    x=np.linspace(-1,1, surfcolor_y.shape[0])
    z=np.linspace(-1,1, surfcolor_y.shape[1])
    z,x=np.meshgrid(z,x)
    y=np.ones(x.shape)
    slice_y=get_the_slice(x,y,z, surfcolor_y)

    y=np.linspace(-1,1, surfcolor_x.shape[0])
    z=np.linspace(-1,1, surfcolor_x.shape[1])
    z,y=np.meshgrid(z,y)
    x=np.ones(z.shape)
    slice_x=get_the_slice(x,y,z, surfcolor_x)

    # sminz, smaxz=get_lims_colors(surfcolor_z)
    # sminy, smaxy=get_lims_colors(surfcolor_y)
    # sminx, smaxx=get_lims_colors(surfcolor_x)
    # vmin=min([sminz, sminy, sminx])
    # vmax=max([smaxz, smaxy, smaxx])

    # slice_z.update(cmin=vmin, cmax=vmax)
    # slice_y.update(cmin=vmin, cmax=vmax)
    # slice_x.update(cmin=vmin, cmax=vmax, showscale=True)
    slice_x.update(showscale=True)

    axis = dict(showbackground=True, 
            backgroundcolor="rgb(230, 230,230)",
            gridcolor="rgb(255, 255, 255)",      
            zerolinecolor="rgb(255, 255, 255)",  
            )


    title = title or 'Slices in volumetric data'
    layout = Layout(
             title=title, 
             width=700,
             height=700,
             scene=Scene(xaxis=XAxis(axis),
                         yaxis=YAxis(axis), 
                         zaxis=ZAxis(axis), 
                         aspectratio=dict(x=1,
                                          y=1, 
                                          z=1
                                         ),
                        )
            )

    fig=Figure(data=Data([slice_z,slice_y,slice_x]), layout=layout)
    py.plot(fig, filename=str_to_filename(title)+'.html', auto_open=False)

def make_slice_plots():
    for scale in [0,1,2]:
        for nu in [0,1,2]:
            for kappa in [0,1,2]:
                cube = tang_psi_window_3D(scale, nu*np.pi/3, kappa*np.pi/3, [7,7,7])
                cube = np.imag(cube)
                title = 'j=%d, nu=%d, kappa=%d' % (scale, nu, kappa)
                pyplot_slices(cube[:,:,3], cube[:,3,:], cube[3,:,:], title=title)

def pyplot_3dscatter(vals, locs, title=None):
    """
    Example usage:
    [vals, locs] = tang_psi_window_3D_flat(1, 1*np.pi/3, 1*np.pi/3, [7,7,7])
    vals = np.imag(vals)
    pyplot_3dscatter(vals, locs)
    """
    
    trace1 = Scatter3d(
        x=locs[:,0],
        y=locs[:,1],
        z=locs[:,2],
        mode='markers',
        marker=dict(
            size=90*np.abs(vals)/vals.max(),
            color=vals,
            colorscale='Hot',
            opacity=0.9
        )
    )

    data = [trace1]
    axis = dict(showbackground=True, 
            backgroundcolor="rgb(230, 230,230)",
            gridcolor="rgb(255, 255, 255)",      
            zerolinecolor="rgb(255, 255, 255)",  
            )
    title = title or '3D Scatter Plot with Colorscaling'
    layout = Layout(
        title=title, 
        width=700,
        height=700,
        # margin=dict(
        #     l=0,
        #     r=0,
        #     b=0,
        #     t=0
        # ),
        scene=Scene(xaxis=XAxis(axis),
                         yaxis=YAxis(axis), 
                         zaxis=ZAxis(axis), 
                        )
    )
    fig = Figure(data=data, layout=layout)
    py.plot(fig, filename=str_to_filename(title)+'.html', auto_open=False)

def make_3dscatter_plots():
    for scale in [0,1,2]:
        for nu in [0,1,2]:
            for kappa in [0,1,2]:
                [vals, locs] = tang_psi_window_3D_flat(scale, nu*np.pi/3, kappa*np.pi/3, [7,7,7])
                vals = np.imag(vals)
                pyplot_3dscatter(vals, locs)
                title = 'j=%d, nu=%d, kappa=%d' % (scale, nu, kappa)
                pyplot_3dscatter(vals, locs, title=title)

class ScrollThruPlot(object):
    """For scrolling through layers of 3d vis
    https://matplotlib.org/2.1.2/gallery/animation/image_slices_viewer.html

    Example usage:

    import matplotlib.pyplot as plt

    X = np.real(cube)
    fig, ax = plt.subplots(1, 1)
    tracker = ScrollThruPlot(ax, X)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    """
    def __init__(self, ax, X, fig):
        self.fig = fig
        self.ax = ax
        self.ax.set_title('Use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = 0

        # if you init with a blank images, all images will be blank
        init_img = np.random.rand(self.X.shape[0], self.X.shape[1])
        self.im = ax.imshow(init_img, cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        # self.im.cla()
        self.im.set_data(self.X[:, :, self.ind] / np.abs(self.X[:, :, self.ind]).max())
        # self.fig.colorbar(self.im)
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

import os
import scipy.io as sio

DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'

def make_dataset_cube_plot():
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right.mat'))
    data = mat_contents['Pavia_center_right'].astype(np.float32)
    data /= np.max(np.abs(data))
    pyplot_cube(data[:,-1,:], data[:,:,51], np.rot90(data[0,:,:]), title='title',resample_factor=1)
    # top, right, left

import windows as win

def display_3dfilter():
    cube = win.fst3d_psi_window_3D(0, 0, 1/7., [7,7,7])
    cube = np.imag(cube)
    pyplot_slices(cube[:,:,3], cube[:,3,:], cube[3,:,:])

if __name__ == '__main__':
    make_dataset_slice_plots()