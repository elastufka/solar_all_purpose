 #######################################
# plot_composite.py
# Erica Lastufka 1/3/2018  

#Description: Plot composite_map. Workaround of SunPy bug #2490
#######################################

#######################################
# Usage:

######################################
import numpy as np
import matplotlib.pyplot as plt
from sunpy.visualization import toggle_pylab, wcsaxes_compat, axis_labels_from_ctype
import astropy.units as u

def composite_plot(composite_map, axes=None, annotate=True,  # pylint: disable=W0613
                           title="SunPy Composite Plot", **matplot_args):
        """Plots the composite map object using matplotlib
        Parameters
        ----------
        axes: `~matplotlib.axes.Axes` or None
            If provided the image will be plotted on the given axes. Else the
            current matplotlib axes will be used.
        annotate : `bool`
            If true, the data is plotted at it's natural scale; with
            title and axis labels.
        title : `str`
            Title of the composite map.
        **matplot_args : `dict`
            Matplotlib Any additional imshow arguments that should be used
            when plotting.
        Returns
        -------
        ret : `list`
            List of axes image or quad contour sets that have been plotted.
        """

        # Get current axes
        if not axes:
            axes = plt.gca()

        if annotate:
            axes.set_xlabel(axis_labels_from_ctype(composite_map._maps[0].coordinate_system[0],
                                                   composite_map._maps[0].spatial_units[0]))
            axes.set_ylabel(axis_labels_from_ctype(composite_map._maps[0].coordinate_system[1],
                                                   composite_map._maps[0].spatial_units[1]))

            axes.set_title(title)
            
        # Define a list of plotted objects
        ret = []
        # Plot layers of composite map
        for m in composite_map._maps:
            # Parameters for plotting
            bl = m._get_lon_lat(m.bottom_left_coord)
            tr = m._get_lon_lat(m.top_right_coord)
            #x_range = list(u.Quantity([bl[0], tr[0]]).to(m.spatial_units[0]).value) #EL that's a non-square aspect ratio...
            y_range = list(u.Quantity([bl[1], tr[1]]).to(m.spatial_units[1]).value)
            x_range = y_range #list(u.Quantity([bl[0], tr[0]]).to(m.spatial_units[0]).value)
            params = {
                "origin": "lower",
                "extent": x_range + y_range,
                "cmap": m.plot_settings['cmap'],
                "norm": m.plot_settings['norm'],
                "alpha": m.alpha,
                "zorder": m.zorder,
            }
            params.update(matplot_args)

            if m.levels is False:
                #ret.append(axes.imshow(m.data, **params))
                if m.mask is None:
                    ret.append(axes.imshow(m.data, **params))
                else:
                    ret.append(axes.imshow(np.ma.array(np.asarray(m.data), mask=m.mask), **params))

            # Use contour for contour data, and imshow otherwise
            if m.levels is not False:
                # Set data with values <= 0 to transparent
                # contour_data = np.ma.masked_array(m, mask=(m <= 0))
                ret.append(axes.contour(m.data, m.levels, **params))
                # Set the label of the first line so a legend can be created
                ret[-1].collections[0].set_label(m.name)

        # Adjust axes extents to include all data
        axes.axis('image')

        # Set current image (makes colorbar work)
        plt.sci(ret[0])
        return ret
