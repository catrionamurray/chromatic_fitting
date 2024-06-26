from ..imports import *

from .lightcurve import *

"""
Example of setting up a PolynomialModel:

def create_new_polynomial_model():
    # create polynomial model:
    p = PolynomialModel(degree=2)
    
    # add empty pymc3 model:
    p.initialize_empty_model()
    
    # add our parameters:
    p.setup_parameters( p_0 = 1.0,
                        p_1 = WavelikeFitted(Uniform,testval=0.01,upper=1,lower=0),
                        p_2 = WavelikeFitted(Uniform,testval=0.01,upper=1,lower=0)
    )
    
    # attach a Rainbow object, r, to the model:
    p.attach_data(r)
    
    # setup the lightcurves for the polynomial model:
    p.setup_lightcurves()
    
    # relate the "actual" data to the model (using a Normal likelihood function)
    p.setup_likelihood()    
    
    # MCMC (NUTS) sample the parameters:
    p.sample(tune=2000, draws=2000, chains=4, cores=4)
    
    # summarize the results:
    p.summarize(round_to=7, fmt='wide')
    
    return p
          
"""


class PolynomialModel(LightcurveModel):
    """
    A polynomial model for the lightcurve.
    """

    def __init__(
        self,
        degree: int,
        independant_variable: str = "time",
        name: str = "polynomial",
        type_of_model: str = "systematic",
        xlims: object = None,
        **kw: object,
    ) -> None:
        """
        Initialize the polynomial model.

        Parameters
        ----------
        degree: the degree of the polynomial
        independant_variable: the independant variable of the polynomial (default = time)
        name: the name of the model (default = "polynomial")
        kw: keyword arguments for initialising the chromatic model
        """
        # only require a constant (0th order) term:
        self.required_parameters = ["p_0"]

        super().__init__(**kw)
        self.degree = degree
        self.independant_variable = independant_variable
        self.xlims = xlims
        self.set_defaults()
        self.metadata = {}
        self.set_name(name)
        self.model = self.polynomial_model

        if type_of_model in allowed_types_of_models:
            self.type_of_model = type_of_model
        else:
            warnings.warn(
                f"{type_of_model} is not a valid type of model. Please select one of: {allowed_types_of_models}"
            )

    def __repr__(self):
        """
        Print the polynomial model.
        """
        return f"<chromatic polynomial model '{self.name}' 🌈>"

    def set_defaults(self):
        """
        Set the default parameters for the model (=0 for each degree).
        """
        for d in range(self.degree + 1):
            try:
                # the | dictionary addition is only in Python 3.9
                self.defaults = self.defaults | {f"p_{d}": 0.0}
            except TypeError:
                # for Python < 3.9 add dictionaries using a different method
                self.defaults = {**self.defaults, **{f"p_{d}": 0.0}}

    def what_are_parameters(self):
        """
        Print a summary of what each parameter is
        # """
        for d in range(self.degree + 1):
            try:
                # the | dictionary addition is only in Python 3.9
                self.parameter_descriptions = self.parameter_descriptions | {f"p_{d}": f"The coefficient of the {d}-degree polynomial"}
            except TypeError:
                # for Python < 3.9 add dictionaries using a different method
                self.parameter_descriptions = {**self.parameter_descriptions, **{f"p_{d}": f"The coefficient of the {d}-degree polynomial"}}

        for k, v in self.parameter_descriptions.items():
            print(f"{k}: {v}")

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

        # if the model has a name then add this to each parameter's name (needed to prevent overwriting parameter names
        # if you combine >1 polynomial model)
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""

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
                # for every wavelength set up a polynomial model
                p = []
                for d in range(self.degree + 1):
                    p.append(self.parameters[f"{name}p_{d}"].get_prior_vector(**kw))

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

                if normalize:
                    # normalise the x values and store the mean/std values in metadata
                    self.metadata["mean_" + self.independant_variable] = np.mean(x)
                    self.metadata["std_" + self.independant_variable] = np.std(x)
                    x = (x - np.mean(x)) / np.std(x)

                    # store the normalised independant variable for later
                    self.independant_variable_normalised = x

                self.normalize = normalize

                if self.xlims is not None:
                    if len(self.xlims) == 2:
                        x[: self.xlims[0]] = 0
                        x[self.xlims[1] :] = 0
                    else:
                        warnings.warn(
                            "You have provided 'xlims' for this polynomial model, however, it needs to be"
                            "in the format xlims=[x1, x2] where x1 and x2 are the min and max indices of the"
                            "data you would like to fit"
                        )

                # compute the polynomial by looping over the coeffs for each degree:

                # to_sub = 0
                poly, initial_guess = [], []
                for i, w in enumerate(data.wavelength):
                    coeff, variable = [], []

                    if len(np.shape(x)) > 1:
                        xi = x[i, :]
                    else:
                        xi = x

                    for d in range(self.degree + 1):
                        if type(p[d]) == float or type(p[d]) == int:
                            coeff.append(p[d])
                            # to_sub += 1
                        elif eval_in_model(p[d].shape) == 1:
                            coeff.append(p[d][0])
                            # to_sub += 1
                        else:
                            coeff.append(p[d][i])# - to_sub])
                        variable.append(xi**d)
                    poly.append(pm.math.dot(coeff, variable))

                    initial_guess.append(eval_in_model(poly[-1]))

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(
                        f"{name}model", pm.math.stack(poly, axis=0)
                    )  # pm.math.sum(poly, axis=0))

                # add the polynomial model to the overall lightcurve:
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.stack(
                        poly, axis=0
                    )  # pm.math.sum(poly, axis=0)
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.stack(
                        poly, axis=0
                    )  # pm.math.sum(poly, axis=0)

                # add the initial guess to the model:
                if f"wavelength_{j}" not in self.initial_guess.keys():
                    self.initial_guess[f"wavelength_{j}"] = np.array(initial_guess)
                else:
                    self.initial_guess[f"wavelength_{j}"] += initial_guess

    def polynomial_model(self, params: dict, i: int = 0) -> np.array:
        """
        Return a polynomial model, given a dictionary of parameters.

        Parameters
        ----------
        params: dictionary with the parameters of the polynomial model
        i: number of the wavelength to fit (default=0)

        Returns
        -------
        np.array: polynomial model with the given parameters
        """
        poly = []

        # if the optimization method is "separate" then extract wavelength {i}'s data
        if self.optimization == "separate":
            data = self.get_data(i)
        else:
            data = self.get_data()

        # for speed extract the variable used during the fit:
        if hasattr(self, "independant_variable_normalised"):
            x = self.independant_variable_normalised
        else:
            # get the independent variable from the Rainbow object:
            x = data.get(self.independant_variable)
            # if the independant variable is time, convert to days:
            if self.independant_variable == "time":
                x = x.to_value("day")
            if self.normalize:
                # normalize:
                x = (x - np.mean(x)) / np.std(x)

        if len(np.shape(x)) > 1:
            if np.shape(x)[0] > 1:
                x = x[i, :]
            else:
                x = x[0, :]

        self.check_and_fill_missing_parameters(params, i)

        try:
            for d in range(self.degree + 1):
                poly.append(params[f"{self.name}_p_{d}"] * (x**d))
            return np.sum(poly, axis=0)
        except KeyError:
            for d in range(self.degree + 1):
                poly.append(params[f"{self.name}_p_{d}_w0"] * (x**d))
            return np.sum(poly, axis=0)

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
