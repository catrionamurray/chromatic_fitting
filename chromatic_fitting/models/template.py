"""
This module serves as a template for creating a new
chromatic_fitting model. If you want to add a new mode a
good process would be to do something like the following:

    1. Copy this `template.py` file into a new file in the
    `models/` directory, ideally with a name that's easy
    to recognize, such as `models/temp.py` (assuming
    `temp` is the name of your model)

    2. Start by finding and replacing `temp` in this
    template with the name of your format.

    3. Edit the `temp_model` function so that it does
    whatever you would like your model to do.

    4. Edit the `models/__init__.py` file to import
    `from .temp import *` for your model to be accessible
    through chromatic_fitting when you want to create a new
    model.

    5. (optional) Submit a pull request to the github
    repository for this package, so that other folks can use
    your handy new model too!
"""

from .lightcurve import *


class tempModel(LightcurveModel):
    """
    A temp model for the lightcurve.
    """

    def __init__(
        self,
        # **ADD ANY EXTRA KEYWORDS THAT YOU WANT TO BE PASSED TO THE MODEL HERE**
        name: str = "temp",
        **kw: object,
    ) -> None:
        """
        Initialize the temp model.

        Parameters
        ----------
        **ADD ANY EXTRA KEYWORDS THAT YOU WANT TO BE PASSED HERE**
        name: the name of the model (default = "temp")
        kw: keyword arguments for initialising the chromatic model
        """
        # **DEFINE THE MINIMUM REQUIRED PARAMETERS TO INITIALISE THIS MODEL**
        list_of_required_parameters_for_the_model_to_run = ["param_1", "param_2"]  # ...
        self.required_parameters = list_of_required_parameters_for_the_model_to_run

        super().__init__(**kw)

        self.set_defaults()
        self.set_name(name)
        self.metadata = {}
        self.model = self.temp_model  # **MAKE SURE TO UPDATE THIS FUNCTION!**

    def __repr__(self):
        """
        Print the temp model.
        """
        return f"<chromatic temp model '{self.name}' 🌈>"

    def set_defaults(self):
        """
        Set the default parameters for the model.
        """
        # **DEFINE THE DEFAULT VALUES FOR (AT LEAST) EACH REQUIRED PARAMETER**
        self.defaults = dict(param_1=0, param_2=1)

    def setup_lightcurves(self, store_models: bool = False, **kwargs):
        """
        Create an temp model, given the stored parameters.
        [This should be run after .attach_data()]

        Parameters
        ----------
        store_models: boolean to determine whether to store the lightcurve model during the MCMC fit

        """

        # if the optimization method is "separate" then loop over each wavelength's model/data
        if self.optimization == "separate":
            models = self._pymc3_model
            datas = [self.get_data(i) for i in range(self.data.nwave)]
        else:
            models = [self._pymc3_model]
            datas = [self.get_data()]
        kw = {"shape": datas[0].nwave}

        # if the model has a name then add this to each parameter's name (needed to prevent overwriting parameter names
        # if you combine >1 polynomial model)
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""

        # if the .every_light_curve attribute (final lc model) is not already present then create it now
        if not hasattr(self, "every_light_curve"):
            self.every_light_curve = {}

        # we can decide to store the LC models during the fit (useful for plotting later, however, uses large amounts
        # of RAM)
        if store_models == True:
            self.store_models = store_models

        # **MAKE SURE TO UPDATE THESE PARAMETERS!**
        param_1, param_2 = [], []
        for j, (mod, data) in enumerate(zip(models, datas)):
            if self.optimization == "separate":
                kw["i"] = j

            with mod:
                # set-up the parameter prior distributions within the model
                # **MAKE SURE TO UPDATE THESE PARAMETERS!**
                param_1.append(self.parameters[f"{name}param_1"].get_prior_vector(**kw))
                param_2.append(self.parameters[f"{name}param_2"].get_prior_vector(**kw))

                y_model = []
                parameters_to_loop_over = {
                    f"{name}param_1": param_1,
                    f"{name}param_2": param_2,
                }

                for i, w in enumerate(data.wavelength):
                    param_i = {}
                    for param_name, param in parameters_to_loop_over.items():
                        if isinstance(self.parameters[param_name], WavelikeFitted):
                            param_i[param_name] = param[j][i]
                        else:
                            param_i[param_name] = param[j]

                    # **FUNCTION TO MODEL - MAKE SURE IT MATCHES self.temp_model()!**
                    function_of_param_i = some_function_of_parameters(param_i)
                    y_model.append(function_of_param_i)

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(f"{name}model", pm.math.stack(y_model, axis=0))

                # add the model to the overall lightcurve:
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.stack(
                        y_model, axis=0
                    )
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.stack(
                        y_model, axis=0
                    )

    def temp_model(self, temp_params: dict, i: int = 0) -> np.array:
        """
        Return a temp model, given a dictionary of parameters.

        Parameters
        ----------
        temp_params: dictionary with the parameters of the temp model
        i: number of the wavelength to fit (default=0)

        Returns
        -------
        np.array: temp model with the given parameters
        """
        temp = []

        # if the optimization method is "separate" then extract wavelength {i}'s data
        if self.optimization == "separate":
            data = self.get_data(i)
        else:
            data = self.get_data()

        self.check_and_fill_missing_parameters(temp_params, i)

        # **FUNCTION TO MODEL - MAKE SURE IT MATCHES self.setup_lightcurves()!**
        temp = some_function_of_parameters(temp_params)

        return temp

    def add_model_to_rainbow(self):
        """
        Add the exponential model to the Rainbow object.
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

        # **ADD MODEL TO rainbow.data_with_model - CHOOSE PLANET OR SYSTEMATIC MODEL (defualt is systematic here)**
        # attach the model to the Rainbow (creating a Rainbow_with_model object)
        r_with_model = data.attach_model(model=model, systematics_model=model)
        # save the Rainbow_with_model for later
        self.data_with_model = r_with_model
