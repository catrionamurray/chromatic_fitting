from ..imports import *

from .lightcurve import *
import warnings


class SpotModel(LightcurveModel):
    """
    A spot model for the lightcurve.
    """

    def __init__(self, name: str = "spot", **kw: object) -> None:
        """
        Initialise the spot model.

        Parameters
        ----------
        name: the name of the model ('spot' by default)
        kw: any keywords to pass to the lightcurve model
        """

        # require the following transit parameters to initialize the model:
        self.required_parameters = [
            "contrast",
            "radius",
            "longitude",
            "latitude",
            "inclination",
            "prot",
        ]

        super().__init__(**kw)
        self.set_defaults()
        self.set_name(name)
        self.model = self.spot_model

    def __repr__(self):
        """
        Print the spot model.
        """
        return f"<chromatic spot model '{self.name}' 🌈>"

    def set_defaults(self):
        """
        Set the default parameters for the model.
        """
        self.defaults = dict(
            contrast=0.1,
            radius=10.0,
            longitude=0.1,
            latitude=0.1,
            inclination=90.0,
            prot=1.0,
        )

    def setup_lightcurves(self, store_models: bool = False):
        """
        Create an `exoplanet` light curve model, given the stored parameters.
        [This should be run after .attach_data()]

        Parameters
        ----------
        store_models: boolean for whether to store the spot model during fitting (for faster
        plotting/access later), default=False
        """
        try:
            import starry
        except ImportError:
            warnings.warn(
                "You don't seem to have starry installed. To use the spot model,"
                "please 'pip install -U starry' before creating this model"
            )
            quit()

        # ensure that attach data has been run before setup_lightcurves
        if not hasattr(self, "data"):
            print("You need to attach some data to this chromatic model!")
            return

        # if the model has a name then add this to each parameter's name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""
            print("No name set for the model.")

        # set up the models and data in a format for looping
        datas, models = self.choose_model_based_on_optimization_method()

        # kw = {"shape": self.get_data().nwave}
        kw = {"shape": datas[0].nwave}

        # if the .every_light_curve attribute (final lc model) is not already present then create it now
        if not hasattr(self, "every_light_curve"):
            self.every_light_curve = {}

        # we can decide to store the LC models during the fit (useful for plotting later, however, uses large amounts
        # of RAM)
        if store_models == True:
            self.store_models = store_models

        # stored_a_r = False

        #         limb_darkening, planet_radius, baseline = [], [], []
        contrast, radius = [], []

        for j, (mod, data) in enumerate(zip(models, datas)):
            if self.optimization == "separate":
                kw["i"] = j

            with mod:

                prot = self.parameters[name + "prot"].get_prior_vector(**kw)

                contrast.append(
                    self.parameters[name + "contrast"].get_prior_vector(**kw)
                )
                radius.append(self.parameters[name + "radius"].get_prior_vector(**kw))
                lat = self.parameters[name + "latitude"].get_prior_vector(**kw)
                lon = self.parameters[name + "longitude"].get_prior_vector(**kw)

                light_curves = []
                for i, w in enumerate(data.wavelength):
                    if isinstance(self.parameters[name + "contrast"], WavelikeFitted):
                        con = contrast[j][i]
                    else:
                        con = contrast[j]

                    if isinstance(self.parameters[name + "radius"], WavelikeFitted):
                        sr = radius[j][i]
                    else:
                        sr = radius[j]

                    # Instantiate the map and add the spot
                    map = starry.Map(ydeg=15)
                    map.inc = self.parameters[name + "inclination"].get_prior_vector(
                        **kw
                    )
                    map.spot(contrast=con, radius=sr, lat=lat, lon=lon)

                    # Compute the flux model
                    flux_model = map.flux(
                        theta=360.0 / prot * data.time.to_value("day")
                    )
                    light_curves.append(flux_model)

                mu = pm.math.stack(light_curves)

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(f"{name}model", mu)

                # add the transit to the final light curve
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = mu  # [i]
                else:
                    self.every_light_curve[f"wavelength_{j}"] += mu  # [i]

                self.model_chromatic_transit_flux = [
                    self.every_light_curve[k] for k in tqdm(self.every_light_curve)
                ]

    def spot_model(self, spot_params: dict, i: int = 0, time: list = None) -> np.array:
        """
        Create a spot model given the passed parameters.

        Parameters
        ----------
        spot_params: A dictionary of parameters to be used in the spot model.
        i: wavelength index
        time: If we don't want to use the default time then the user can pass a time array on which to calculate the model

        Returns
        -------
        object
        """
        try:
            import starry
        except ImportError:
            warnings.warn(
                "You don't seem to have starry installed. To use the spot model,"
                "please 'pip install -U starry' before creating this model"
            )
            quit()

        if self.optimization == "separate":
            data = self.get_data(i)
        else:
            data = self.get_data()

        # if the model has a name then add this to each parameter"s name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""

        if time is None:
            time = data.time.to_value("day")

        self.check_and_fill_missing_parameters(spot_params, i)

        # Instantiate the map and add the spot
        map = starry.Map(ydeg=15)
        map.inc = spot_params[f"{name}inclination"]
        map.spot(
            contrast=spot_params[f"{name}contrast"],
            radius=spot_params[f"{name}radius"],
            lat=spot_params[f"{name}latitude"],
            lon=spot_params[f"{name}longitude"],
        )

        # Compute the flux model
        lc = map.flux(theta=360.0 / spot_params[f"{name}prot"] * time).eval()

        return lc

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
        r_with_model = data.attach_model(model=model, systematics_model=model)
        # save the Rainbow_with_model for later
        self.data_with_model = r_with_model
