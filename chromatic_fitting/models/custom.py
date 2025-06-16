from ..imports import *

from .lightcurve import *

"""
Example of setting up a CustomModel:

def create_new_custom_model():
    # create custom model:
    x = np.linspace(0,1,100)
    p = CustomModel(y=np.sin(x))

"""

class CustomModel(LightcurveModel):
    """
    A custom model for the lightcurve (there is no fitting involved).
    """

    def __init__(
            self,
            y,
            independant_variable: str = "time",
            name: str = "polynomial",
            type_of_model: str = "systematic",
            xlims: object = None,
            **kw: object,
    ) -> None:
        """
        Initialize the custom model.

        Parameters
        ----------
        independant_variable: the independant variable of the polynomial (default = time)
        name: the name of the model (default = "polynomial")
        kw: keyword arguments for initialising the chromatic model
        """
        self.required_parameters = []

        super().__init__(**kw)
        self.independant_variable = independant_variable
        self.xlims = xlims
        self.set_defaults()
        self.metadata = {}
        self.set_name(name)
        self.y = y
        self.model = self.custom_model

        if type_of_model in allowed_types_of_models:
            self.type_of_model = type_of_model
        else:
            warnings.warn(
                f"{type_of_model} is not a valid type of model. Please select one of: {allowed_types_of_models}"
            )

    def __repr__(self):
        """
        Print the custom model.
        """
        return f"<chromatic custom model '{self.name}' 🌈>"

    def set_defaults(self):
        """
        There are no default parameters for a CustomModel
        """
        self.defaults = {}

    def setup_lightcurves(
            self, store_models: bool = False, normalize: bool = True, **kwargs
    ):
        """
        Create a polynomial model, given the stored parameters.
        [This should be run after .attach_data()]

        Parameters
        ----------
        store_models: boolean to determine whether to store the lightcurve model during the MCMC fit

        """

        # if the optimization method is "separate" then loop over each wavelength's model/data
        datas, models = self.choose_model_based_on_optimization_method()
        kw = {"shape": datas[0].nwave}

        # if the .every_light_curve attribute (final lc model) is not already present then create it now
        if not hasattr(self, "every_light_curve"):
            self.every_light_curve = {}
        if not hasattr(self, "initial_guess"):
            self.initial_guess = {}

        # we can decide to store the LC models during the fit (useful for plotting later, however, uses large amounts
        # of RAM)
        if store_models == True:
            self.store_models = store_models

        for j, (mod, data) in enumerate(zip(models, datas)):
            if self.optimization == "separate":
                kw["i"] = j

            with mod:

                # get the independent variable from the Rainbow object:
                x = data.get(self.independant_variable)

                # if the independant variable is time, convert to days:
                if self.independant_variable == "time":
                    x = x.to_value("day")
                else:
                    try:
                        x = x.to_value()
                    except AttributeError:
                        pass

                cust, initial_guess = [], []
                for i, w in enumerate(data.wavelength):
                    cust.append(np.array(self.y))
                    initial_guess.append(np.array(self.y))
                if np.shape(cust)[0] == 1:
                    cust = cust[0]
                    initial_guess = initial_guess[0]

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(
                        f"{self.name}_model", pm.math.stack(cust)
                    )

                    # add the custom model to the overall lightcurve:
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.stack(cust)
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.stack(cust)

                # add the initial guess to the model:
                if f"wavelength_{j}" not in self.initial_guess.keys():
                    self.initial_guess[f"wavelength_{j}"] = np.array(initial_guess)
                else:
                    self.initial_guess[f"wavelength_{j}"] += initial_guess

    def add_model_to_rainbow(self):
        """
        Add the polynomial model to the Rainbow object.
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

    def custom_model(self, params=None, i=0, **kw):
        return self.y