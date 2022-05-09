from .imports import *
from .spectrum import *
from chromatic import SimulatedRainbow

class star:
    def __init__(self, logg, M_H, Teff):
        '''
        Create star object using literature parameter values

        Parameters
        ----------
        logg : float
            Gravity
        M_H : float
            Metallicity
        Teff : float
            Effective Temperture (K)
        '''

        self.logg = logg
        self.M_H = M_H
        self.Teff = Teff

class synthetic_planet(PlanetarySpectrumModel):

    def add_stellar_params(self,star_params):
        '''
        Add stellar parameters to the PlanetarySpectrumModel defined in spectrum.py

        Parameters
        ----------
        star_params : dict
            Dictionary of stellar parameters must currently contain gravity, "logg", metallicity, "M_H", and effective
            temperature, "Teff".
        '''
        self.stellar = star(logg=star_params['logg'],M_H=star_params['M_H'],Teff=star_params['Teff'])


    def generate_ld_coeffs(self,mode,dirsen,ld_eqn,ld_model):
        '''
        Generate synthetic Limb-Darkening coefficients for a range of wavelengths using ExoTiC-LD

        Parameters
        ----------
        mode : str
            The instrument/wavelength band defined by ExoTiC-LD - several available for JWST.
        dirsen : str
            The directory path where the stellar model grid and instrument information files required by ExoTiC are stored
        ld_eqn : str
            The equation used to calculate the limb-darkening coeffs, options "linear", "quadratic", "nonlinear" and
            "threeparam" (NOTE: There is a current issue open in ExoTiC to add a selection criteria so that only the
            desired coefficients are returned.)
        ld_model : str
            Either use "1D" or "3D" stellar model grid
        '''

        ld_coeffs,mask = [],[]
        prev_ld = 0.0

        # For ExoTiC we need a range of wavelengths - create bins around each value (not ideal)
        for w in range(len(self.table["wavelength"])):
            # wsdata = [self.table["wavelength_lower"][w],self.table["wavelength_upper"][w]]
            wsdata = np.arange(self.table["wavelength_lower"][w], self.table["wavelength_upper"][w],0.001)
            print("Wavelength range: ", wsdata)

            # * * * * * * * *
            # Use ExoTiC-LD to calculate wavelength-dependent LD coeffs
            result = limb_dark_fit(mode, np.array(wsdata) * 10000, self.stellar.M_H, self.stellar.Teff, self.stellar.logg, dirsen, ld_model=ld_model)
            # * * * * * * * *

            # If all zeros are returned then ignore (happens when we're outside the wavelength range defined by 'mode')
            if np.all(np.array(result) == 0.0):
                if prev_ld == 0.0:
                    print("Outside " + mode + " wavelength range!\n")
                    mask.append(1)
                    ld_coeffs.append(np.nan)
                else:
                    print("There may be an issue with grid spacing!\n")
                    mask.append(1)
                    ld_coeffs.append(np.nan)
                    prev_ld = 0.0
            else:
                prev_ld = result[0]
                if ld_eqn == "linear":
                    print("Using linear LD equation\n")
                    ld_coeffs.append(result[0])
                    mask.append(0)

                elif ld_eqn == "quadratic":
                    print("Using quadratic LD equation\n")
                    ld_coeffs.append(result[9:11])
                    mask.append(0)

                elif ld_eqn == "nonlinear":
                    print("Using non-linear LD equation\n")
                    ld_coeffs.append(result[1:5])
                    mask.append(0)

                elif ld_eqn == "threeparam":
                    print("Using three-paramter LD equation\n")
                    ld_coeffs.append(result[6:9])
                    mask.append(0)

                else:
                    print("No valid LD equation method chosen!\n")

        self.modemask = np.array(mask)
        self.ld_coeffs = np.array(ld_coeffs)
        self.ld_eqn = ld_eqn
        self.ld_model = ld_model

    def plotmask(self,ax=None, **kw):
        '''
        Plot the masked model.

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

        plot_kw = dict(alpha=0.75, linewidth=2, label=self.label + " (masked)")
        plot_kw.update(**kw)
        plt.plot(self.table['wavelength'][self.modemask==0], self.table['depth'][self.modemask==0], color='orange', **plot_kw)
        plt.xlabel("Wavelength (micron)")
        plt.ylabel("Depth (unitless)")
        return ax

    def plotLDcoeffs(self,ax=None, **kw):
        '''
        Plot the synthetic LD coeffs.

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

        num_ld_coeffs = len(self.ld_coeffs[self.modemask==0][0])
        # print(num_ld_coeffs, len(self.ld_coeffs[~np.isnan(self.ld_coeffs)][0]))

        for ld in range(num_ld_coeffs):
            plot_kw = dict(alpha=0.75, linewidth=2, label="LD Coeff " + str(ld))
            plot_kw.update(**kw)
            extract_coeff = [l[ld] for l in self.ld_coeffs[self.modemask==0]]
            plt.plot(self.table['wavelength'][self.modemask==0], extract_coeff, **plot_kw)

        plt.title(self.ld_eqn + " Limb-Darkening")
        plt.xlabel("Wavelength (micron)")
        plt.ylabel("Limb-Darkening Coeff")
        return ax



def semi_major_axis(per,M,R):
    '''
        Use Kepler's laws to calculate the semi-major axis from period, mass and radius.

        Parameters
        ----------
        per : float
            Orbital period (days).
        M : float
            Stellar mass (Solar masses)
        R : float
            Stellar radius (Solar radii)

        Returns
        ----------
        a_radii : float
            a/R*, semi-major axis normalised by the Solar radius.

        '''

    per_s = per * 24 * 60 * 60 * u.s
    R_sun = R * 696340000 * u.m
    M_sun = M * 1.989e30 * u.kg

    a = ((per_s)**2 * G * M_sun/ (4 * math.pi**2))**(1./3) # in units of m
    a_radii = a / R_sun

    return a_radii

def generate_spectrum_ld(wavelength, depth, star_params,planet_params,dirsen,mode='NIRCam_F322W2',ld_eqn='quadratic',ld_model='1D',plot_model=True):
    '''
    Generate the wavelength-dependent limb-darkening coefficients to generate a synthetic spectrum.

    Parameters
    ----------
    wavelength : np.array or list
        The centres of the wavelength bins to calculate the LD coeffs over.
    depth : np.array or list
        The transit depths as a function of wavelength.
    star_params : dict
        Stellar parameters from user. For now, must contain: M_H, teff, logg.
    planet_params : dict
        Synthetic planet parameters from user.
    dirsen : str
        The directory path where the stellar model grid and instrument information files required by ExoTiC are stored.
    mode : str
            The instrument/wavelength band defined by ExoTiC-LD - several available for JWST.
    ld_eqn : str
        The equation used to calculate the limb-darkening coeffs, options "linear", "quadratic", "nonlinear" and
        "threeparam" (NOTE: There is a current issue open in ExoTiC to add a selection criteria so that only the
        desired coefficients are returned).
    ld_model : str
         Either use "1D" or "3D" stellar model grid.
    plot_model : bool
        Plot the synthetic spectrum and the LD coeffs against wavelength

    Returns
    ----------
    model : synthetic_planet
        Synthetic planet object to inject into Rainbow object.

    '''

    table = Table(dict(wavelength=wavelength, depth=depth), meta=planet_params)
    model = synthetic_planet(table=table, label='injected model')
    model.add_stellar_params(star_params)
    model.generate_ld_coeffs(mode,dirsen,ld_eqn,ld_model)

    # plot model provided
    if plot_model==True:
        fig,(ax1,ax2) = plt.subplots(nrows=2,sharex=True,figsize=(8, 6), dpi=300)
        ax1 = model.plot(ax=ax1)
        ax1 = model.plotmask(ax=ax1)
        ax2 = model.plotLDcoeffs(ax=ax2)
        ax1.legend(frameon=False)
        ax2.legend(frameon=False)
        plt.show()
        plt.close()



    return model

def inject_spectrum(model,snr=100,dt=1,res=50,planet_params={}):
    '''
    Inject the synthetic model into chromatic Rainbow

    Parameters
    ----------
    model : synthetic_planet
        Synthetic planet object with wavelength-varying limb-darkening coefficients, as defined in generate_spectrum_ld
    snr : int
        Signal-to-noise ratio of the resulting light curve.
    dt : int
        Time resolution of resulting light curve.
    res : int
        Resolution of spectrum
    planet_params : dict
        Dictionary of planet parameters (defined in Batman) to pass to chromatic injection code

    Returns
    ----------
    r : SimulatedRainbow
        Simulated Rainbow object from chromatic without the injected transit.
    i : SimulatedRainbow
        Simulated Rainbow object (r) with injected transit.

    '''

    modemask = model.modemask

    planet_params.update({"limb_dark": model.ld_eqn, 'u':list(model.ld_coeffs[modemask==0])})

    r = SimulatedRainbow(
        signal_to_noise=snr,
        dt=dt * u.minute,
        wavelength=np.array(model.table["wavelength"][modemask==0])*u.micron,
        R=res
    )
    i = r.inject_transit(
        planet_radius=np.array(model.table['depth'][modemask==0]),
        planet_params= planet_params #{"limb_dark": model.ld_eqn, 'u':list(model.ld_coeffs[modemask==0])} #planet_params
    )

    return r,i
