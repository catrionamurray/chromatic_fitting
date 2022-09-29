from ..imports import *

from .lightcurve import *
import warnings

"""
Example of setting up a TransitModel:

def create_new_transit_model():
    # create transit model:
    t = TransitModel()
    
    # add empty pymc3 model:
    t.initialize_empty_model()
    
    # add our parameters:
    t.setup_parameters(
                      period=1, # a fixed value!
                       epoch=Fitted(Uniform,lower=-0.05,upper=0.05), # one fitted value across all wavelengths
                       stellar_radius = Fitted(Uniform, lower=0.8, upper=1.2,testval=1),
                       stellar_mass =Fitted(Uniform, lower=0.8, upper=1.2,testval=1),
                       radius_ratio=WavelikeFitted(Normal, mu=0.5, sigma=0.05), # a value fitted for every wavelength!
                       impact_parameter=Fitted(ImpactParameter,ror=0.15,testval=0.44),
                       limb_darkening=WavelikeFitted(QuadLimbDark,testval=[0.05,0.35]),
                        baseline = WavelikeFitted(Normal, mu=1.0, sigma=0.05), # I will keep this fixed for when we add a polynomial model!
                    )
    
    # attach a Rainbow object, r, to the model:
    t.attach_data(r)
    
    # setup the lightcurves for the transit model:
    t.setup_lightcurves()
    
    # relate the "actual" data to the model (using a Normal likelihood function)
    t.setup_likelihood()    
    
    # MCMC (NUTS) sample the parameters:
    t.sample(tune=2000, draws=2000, chains=4, cores=4)
    
    # summarize the results:
    t.summarize(round_to=7, fmt='wide')
    
    return t
                
"""


class TransitModel(LightcurveModel):
    """
    A transit model for the lightcurve.
    """

    def __init__(self, name: str = "transit", **kw: object) -> None:
        """
        Initialise the transit model.

        Parameters
        ----------
        name: the name of the model ('transit' by default)
        kw: any keywords to pass to the lightcurve model
        """

        # require the following transit parameters to initialize the model:
        self.required_parameters = [
            "stellar_radius",
            "stellar_mass",
            "radius_ratio",
            "period",
            "epoch",
            "baseline",
            "impact_parameter",
            "limb_darkening",
        ]

        super().__init__(**kw)
        self.set_defaults()
        self.set_name(name)

    def set_defaults(self):
        """
        Set the default parameters for the model.
        """
        self.defaults = dict(
            stellar_radius=1.0,
            stellar_mass=1.0,
            radius_ratio=1.0,
            period=1.0,
            epoch=0.0,
            baseline=1.0,
            impact_parameter=0.5,
            eccentricity=0.0,
            omega=np.pi / 2.0,
            limb_darkening=[0.2, 0.2],
        )

    def __repr__(self):
        """
        Print the transit model.
        """
        return f"<chromatic transit model '{self.name}' 🌈>"

    def set_name(self, name: str):
        """
        Set the name of the model.

        Parameters
        ----------
        name: the name of the model
        """
        self.name = name

    def setup_orbit(self):
        """
        Create an `exoplanet` orbit model, given the stored parameters.
        [If run manually it should be run after .setup_parameters()]

        """

        # if the optimization method is separate wavelengths then set up for looping
        if self.optimization == "separate":
            models = self.pymc3_model
        else:
            models = [self.pymc3_model]

        # if the model has a name then add this to each parameter"s name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""
            print("No name set for the model.")

        self.orbit = []
        for j, mod in enumerate(models):
            with mod:

                # Set up a Keplerian orbit for the planets (for now without eccentricity)
                orbit = xo.orbits.KeplerianOrbit(
                    period=self.parameters[name + "period"].get_prior(j),
                    t0=self.parameters[name + "epoch"].get_prior(j),
                    b=self.parameters[name + "impact_parameter"].get_prior(j),
                    # ecc=self.parameters[name + "eccentricity"].get_prior(j),
                    # omega=self.parameters[name + "omega"].get_prior(j),
                    r_star=self.parameters[name + "stellar_radius"].get_prior(j),
                    m_star=self.parameters[name + "stellar_mass"].get_prior(j),
                )

                # store a separate orbit for each model if we are optimizing wavelengths separately
                if self.optimization == "separate":
                    self.orbit.append(orbit)
                else:
                    self.orbit = orbit

    def setup_lightcurves(self, store_models: bool = False):
        """
        Create an `exoplanet` light curve model, given the stored parameters.
        [This should be run after .attach_data()]

        Parameters
        ----------
        store_models: boolean for whether to store the transit model during fitting (for faster
        plotting/access later), default=False
        """

        # ensure that attach data has been run before setup_lightcurves
        if not hasattr(self, "data"):
            print("You need to attach some data to this chromatic model!")
            return

        # ensure that setup_orbit has been run before setup_lightcurves
        if not hasattr(self, "orbit"):
            self.setup_orbit()

        # if the model has a name then add this to each parameter's name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""
            print("No name set for the model.")

        # set up the models, data and orbits in a format for looping
        if self.optimization == "separate":
            models = self.pymc3_model
            datas = [self.get_data(i) for i in range(self.data.nwave)]
            orbits = self.orbit
        else:
            models = [self.pymc3_model]
            datas = [self.get_data()]
            orbits = [self.orbit]

        # if the .every_light_curve attribute (final lc model) is not already present then create it now
        if not hasattr(self, "every_light_curve"):
            self.every_light_curve = {}

        # we can decide to store the LC models during the fit (useful for plotting later, however, uses large amounts
        # of RAM)
        if store_models == True:
            self.store_models = store_models

        for j, (mod, data, orbit) in enumerate(zip(models, datas, orbits)):
            with mod:
                for i, w in enumerate(data.wavelength):
                    # for each wavelength create a quadratic limb-darkening lightcurve model from Exoplanet
                    limb_darkening = self.parameters[name + "limb_darkening"].get_prior(
                        j + i
                    )

                    planet_radius = self.parameters[name + "radius_ratio"].get_prior(
                        j + i
                    ) * self.parameters[name + "stellar_radius"].get_prior(j + i)

                    # extract light curve from Exoplanet model at given times
                    light_curves = xo.LimbDarkLightCurve(
                        limb_darkening
                    ).get_light_curve(
                        orbit=orbit,
                        r=planet_radius,
                        t=list(data.time.to_value("day")),
                    )

                    # store Deterministic parameter {a/R*} for use later
                    Deterministic(
                        f"{name}a_R*_{i + j}",
                        orbit.a
                        / self.parameters[name + "stellar_radius"].get_prior(j + i),
                    )

                    # calculate the transit + flux (out-of-transit) baseline model
                    mu = pm.math.sum(light_curves, axis=-1) + (
                        self.parameters[name + "baseline"].get_prior(j + i)
                    )

                    # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                    # later:
                    if self.store_models:
                        Deterministic(f"{name}model_w{i + j}", mu)

                    # add the transit to the final light curve
                    if f"wavelength_{j + i}" not in self.every_light_curve.keys():
                        self.every_light_curve[f"wavelength_{j + i}"] = pm.math.sum(
                            light_curves, axis=-1
                        ) + (self.parameters[name + "baseline"].get_prior(j + i))
                    else:
                        self.every_light_curve[f"wavelength_{j + i}"] += pm.math.sum(
                            light_curves, axis=-1
                        ) + (self.parameters[name + "baseline"].get_prior(j + i))

                self.model_chromatic_transit_flux = [
                    self.every_light_curve[k] for k in tqdm(self.every_light_curve)
                ]

    def transit_model(self, transit_params: dict, i: int = 0) -> np.array:
        """
        Create a transit model given the passed parameters.

        Parameters
        ----------
        transit_params: A dictionary of parameters to be used in the transit model.
        i: wavelength index

        Returns
        -------
        object
        """

        if self.optimization == "separate":
            data = self.get_data(i)
        else:
            data = self.get_data()

        # if the model has a name then add this to each parameter"s name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""
        # name = self.name

        orbit = xo.orbits.KeplerianOrbit(
            period=transit_params[f"{name}period"],
            t0=transit_params[f"{name}epoch"],
            b=transit_params[f"{name}impact_parameter"],
            # ecc=transit_params["eccentricity"],
            # omega=transit_params["omega"],
            r_star=transit_params[f"{name}stellar_radius"],
            m_star=transit_params[f"{name}stellar_mass"],
        )

        ldlc = (
            xo.LimbDarkLightCurve(transit_params[f"{name}limb_darkening"])
            .get_light_curve(
                orbit=orbit,
                r=transit_params[f"{name}radius_ratio"]
                * transit_params[f"{name}stellar_radius"],
                t=list(data.time.to_value("day")),
            )
            .eval()
        )
        return ldlc.transpose()[0] + transit_params[f"{name}_baseline"]

    def plot_orbit(self, timedata: object = None):
        """
        Plot the orbit model (to check that the planet is transiting at the times we've set).

        Parameters
        ----------
        timedata: An array of times to plot the orbit at, only used if .attach_data() hasn't been
        run yet (default=None)

        """

        # If the data hasn't been attached yet, then use the timedata passed to the function
        if not hasattr(self, "data"):
            if timedata is None:
                warnings.warn(
                    "No data attached to this object and no time data provided. Plotting orbit will not work."
                )
                print(
                    "No data attached to this object and no time data provided. Plotting orbit will not work."
                )
                return
        else:
            timedata = self.data.time

        # if the optimization method is separate wavelengths then set up for looping
        if self.optimization == "separate":
            models = self.pymc3_model
            orbits = self.orbit
        else:
            models = [self.pymc3_model]
            orbits = [self.orbit]

        # plot the orbit
        for j, (mod, orbit) in enumerate(zip(models, orbits)):
            with mod:
                # find the x,y,z position of the planet at each timestamp relative to the centre of the star
                x, y, z = [
                    eval_in_model(bla, point=mod.test_point)
                    for bla in orbit.get_planet_position(timedata)
                ]
                plt.figure(figsize=(10, 3))
                theta = np.linspace(0, 2 * np.pi)
                plt.fill_between(np.cos(theta), np.sin(theta), color="gray")
                plt.scatter(x, y, c=timedata)
                plt.axis("scaled")
                plt.ylim(-1, 1)
                plt.show()
                plt.close()

    def get_model(self, as_dict: bool = True, as_array: bool = False):
        """
        Return the 'best-fit' model from the summary table as a dictionary or as an array

        Parameters
        ----------
        as_dict: boolean whether to return the model as a dictionary (with keys indexing the wavelength)
        as_array: boolean whether to return the model as an array

        Returns
        -------
        object: transit model for each wavelength (either a dict or array)
        """

        # if the optimization method is "separate" then loop over each wavelength's data
        if self.optimization == "separate":
            datas = [self.get_data(i) for i in range(self.data.nwave)]
        else:
            datas = [self.get_data()]

        # if we decided to store the LC model extract this now, otherwise generate the model:
        if self.store_models:
            return LightcurveModel.get_model(as_dict=as_dict, as_array=as_array)
        else:
            model = {}
            # generate the transit model from the best fit parameters for each wavelength
            for i, data in enumerate(datas):
                transit_params = self.extract_from_posteriors(self.summary, i)
                model_i = self.transit_model(transit_params, i)
                model[f"w{i}"] = model_i
            if as_array:
                # return a 2D array (one row for each wavelength)
                return np.array(list(model.values()))
            elif as_dict:
                # return a dict (one key for each wavelength)
                return model

    def add_model_to_rainbow(self):
        """
        Add the transit model to the Rainbow object.
        """
        # if we decided to flag outliers then flag these in the final model
        if self.outlier_flag:
            data = self.data_without_outliers
        else:
            data = self.data

        # if optimization method is "white_light" then extract the white light curve
        if self.optimization == "white_light":
            data = self.white_light

        # extract model as an array
        model = self.get_model(as_array=True)
        # attach the model to the Rainbow (creating a Rainbow_with_model object)
        r_with_model = data.attach_model(model=model, planet_model=model)
        # save the Rainbow_with_model for later
        self.data_with_model = r_with_model
