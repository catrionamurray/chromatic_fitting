from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import warnings

class PlanetarySpectrum:
    @property
    def wavelength(self):
        return self.table['wavelength'].quantity

    @property
    def depth(self):
        return self.table['depth'].quantity

    @property
    def uncertainty(self):
        return self.table['uncertainty'].quantity
    
    def __init__(self, table=None, label=None):
        '''
        Initialize planetary spectrum object.

        Parameters
        ----------
        table : astropy.table.Table
            A table of depths (or other wavelength-dependent features).
            It should contain at least:
                + `wavelength` should represent the central wavelength of the 
                   wavelength bin. Alternatively, there could be two columns 
                   labeled `wavelength_lower` and `wavelength_upper` to represent 
                   the lower and upper bounds of each wavelength bin. The units 
                   should be in microns.
                + `depth` should be the transit depth $(R_p/R_\star)^2$ or the 
                   eclipse depth ($F_p/F_\star$). This quantity should be unitless; 
                   for example, a transit depth of 1% should be written as `0.01`.
                + `uncertainty` should be the uncertainty on the depth (if this 
                   is data). This quantity should have the same units as depth 
                   (so also be unitless).
             Planet parameters can/should be included as `meta` in this 
             initializing astropy table.
         label : str
             A string labeling this planet spectrum. It might appear in the 
             the display name for the object, plot legends, filenames, ...             
         '''

        # store the original data inputs
        self.table = Table(table)

        # store additional information that might be handy
        self.label = label        
        
        # make sure the data format works
        self._validate_data()


    def _validate_data(self):
        '''
        Make sure the data table has the right format.
        '''
        
        # validate each core component
        self._validate_wavelengths()
        self._validate_depths()
        self._validate_uncertainties()
        
    def _validate_wavelengths(self):
        '''
        Make sure wavelengths are usable.
        '''
            
        # set centers from edges, if necessary
        if "wavelength" not in self.table.columns:
            self.table["wavelength"] = 0.5 * (self.table["wavelength_lower"] + self.table["wavelength_upper"])

        # set edges from centers, if necessary
        if ("wavelength_lower" not in self.table.columns) and ("wavelength_upper" not in self.table.columns):
            bin_widths = np.gradient(self.table["wavelength"])
            self.table["wavelength_lower"] = self.table["wavelength"] - bin_widths / 2
            self.table["wavelength_upper"] = self.table["wavelength"] + bin_widths / 2    

        # make sure the units are good
        for k in ['wavelength', 'wavelength_lower', 'wavelength_upper']:
            try:
                self.table[k] = self.table[k].to(u.micron)
                print(f'units worked for {k}')
            except (AttributeError, u.UnitConversionError):
                self.table[k] = self.table[k]*u.micron
                print(f'units needed to be fudged for {k}')
                warnings.warn(f'🌈 Assuming units for {k} are micron.')
                    
        assert('wavelength' in self.table.columns)

    def _validate_depths(self):
        '''
        Make sure depths are usable.
        '''        
        if np.all(self.depth > 1):
            messages = '''
            🪐 All depths are >1, implying the planet is 
            bigger than the star. Depths, should be unitless,
            so a 1% transit should have a depth of 0.01.
            '''
            warnings.warn(message)
  
    def _validate_uncertainties(self):
        '''
        Make sure uncertainties are usable.
        '''        
        pass
            
    def __repr__(self):
        if 'uncertainty' in self.table.columns:
            extra = ' with uncertainties!'
        else:
            extra = ''
        return f'<🪐PlanetarySpectrum({len(self.wavelength)}w{extra})>'
    
class PlanetarySpectrumModel(PlanetarySpectrum):
    
    def plot(self, ax=None, **kw):
        '''
        Plot the model.
        
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes into which the plot should be drawn.
        kw : dict
            Extra keywords will be passed along to `plt.plot`
        '''
        if ax is None:
            plt.figure(figsize=(8, 6), dpi=300)
            ax = plt.gca()
        else:
            plt.sca(ax)
            
        plot_kw = dict(alpha=0.5, linewidth=2, label=self.label)
        plot_kw.update(**kw)
        plt.plot(self.table['wavelength'], self.table['depth'], **plot_kw)
        plt.xlabel("Wavelength (micron)")
        plt.ylabel("Depth (unitless)")
        return ax
    
class PlanetarySpectrumData(PlanetarySpectrum):
    
    def plot(self, ax=None, **kw):
        """
        Plot some planetary features.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes into which the plot should be drawn.
        """
#         table = Table(self)

        lower = self.table["wavelength"] - self.table["wavelength_lower"]
        upper = self.table["wavelength_upper"] - self.table["wavelength"]

        errorkw = dict(marker="o", linewidth=0, elinewidth=1, label=self.label)
        errorkw.update(**kw)

        if ax is None:
            plt.figure(figsize=(8, 6), dpi=300)
            ax = plt.gca()
        else:
            plt.sca(ax)
        plt.errorbar(
            self.table["wavelength"],
            self.table["depth"],
            yerr=self.table["uncertainty"],
            xerr=[lower, upper],
            **errorkw,
        )
        plt.xlabel("Wavelength (micron)")
        plt.ylabel("Depth (unitless)")
        return ax