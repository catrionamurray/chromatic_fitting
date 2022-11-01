import pymc3 as pm
import numpy as np


class Parameter:
    """
    Parameter objects manage model parameters, which may be either fixed or fitted.
    """

    def set_name(self, name):
        self.name = name


class Fixed(Parameter):
    """
    Fixed parameters are just constant values or arrays
    that are shared across all wavelengths.
    """

    def __init__(self, value, **inputs):
        self.value = value

    def get_prior(self, *args, **kwargs):
        """
        Get the "prior", which is just a value.

        Returns
        -------
        prior : float, array
            The value or array for this parameter.
        """
        return self.value

    def get_prior_vector(self, *args, **kwargs):
        return self.get_prior(self, *args, **kwargs)

    def __repr__(self):
        return f"<🧮 Fixed | {self.value} 🧮>"


class WavelikeFixed(Fixed):
    """
    WavelikeFixed parameters are just constant values or arrays
    that are unique to each wavelength.
    """

    def __init__(self, values, **inputs):
        self.values = values

    def get_prior(self, i, *args, **kwargs):
        """
        Get the "prior", which is just a value, for this wavelength.

        Parameters
        ----------
        i : int
            The index of the wavelength associated with this prior.

        Returns
        -------
        prior : float, array
            The value or array for this parameter at this wavelength.
        """
        return self.values[i]

    def get_prior_vector(self, i=None, shape=1, *args, **kwargs):
        if i is None:
            return [self.get_prior(i, *args, **kwargs) for i in range(shape)]
        else:
            return self.get_prior(i, *args, **kwargs)

    def __repr__(self):
        return f"<🧮 WavelikeFixed | one value for each wavelength ({len(self.values)} elements)🧮>"


class Fitted(Parameter):
    """
    Fitted parameters are allowed to be fit, characterized by
    some prior distribution. They are shared across wavelengths.
    """

    def __init__(self, distribution, **inputs):
        self.distribution = distribution
        self.inputs = inputs

    def set_name(self, name):
        self.name = name
        self.inputs["name"] = name

    def generate_pymc3(self, *args, **kwargs):
        """
        Generate a PyMC3 prior.

        Parameters
        ----------
        kw : dict
            All keyword arguments will be ignored.
        """
        self._pymc3_prior = self.distribution(**self.inputs)
        return self._pymc3_prior

    def get_prior(self, *args, **kwargs):
        """
        Get the PyMC3 prior.

        Returns
        -------
        prior : PyMC3 distribution
            The prior for this parameter for this wavelength
        """
        try:
            return self._pymc3_prior
        except AttributeError:
            return self.generate_pymc3()

    def generate_pymc3_vector(self, *args, **kwargs):
        """
        Generate a PyMC3 prior.

        Parameters
        ----------
        kw : dict
            All keyword arguments will be ignored.
        """
        if "shape" not in self.inputs:
            self.inputs["shape"] = 1
        self._pymc3_prior = self.distribution(**self.inputs)
        return self._pymc3_prior

    def get_prior_vector(self, *args, **kwargs):
        """
        Get the PyMC3 prior.

        Returns
        -------
        prior : PyMC3 distribution
            The prior for this parameter for this wavelength
        """
        try:
            return self._pymc3_prior
        except AttributeError:
            return self.generate_pymc3_vector()

    def clear_prior(self, *args, **kwargs):
        """
        Clear the stored PyMC3 prior.
        """
        try:
            delattr(self, "_pymc3_prior")
            print(f"Cleared {self.name} prior")
        except AttributeError:
            pass

    def __repr__(self):
        distribution_name = self.distribution.__name__
        inputs_as_string = ", ".join([f"{k}={repr(v)}" for k, v in self.inputs.items()])
        return f"<🧮 Fitted {distribution_name}({inputs_as_string}) 🧮>"


class WavelikeFitted(Fitted):
    """
    WavelikeFitted parameters are allowed to be fit, characterized by
    some prior distribution. They are unique to each wavelength.
    """

    def __init__(self, distribution, **inputs):
        self.distribution = distribution
        self.inputs = inputs
        self._pymc3_priors = {}

    def label(self, i):
        return f"{self.inputs['name']}_w{i}"

    def generate_pymc3(self, i, *args, **kwargs):
        """
        Generate a PyMC3 prior for wavelength i.

        Parameters
        ----------
        i : int
            The index of the wavelength associated with this prior.
        kw : dict
            All keyword arguments will be ignored.
        """
        inputs = self.inputs.copy()
        # if a different prior is passed for every wavelength then we use only the one for wavelength i
        for k, v in self.inputs.items():
            if len(np.shape(v)) > 1:
                inputs[k] = v[i]

        inputs = dict(**inputs)
        inputs["name"] = self.label(i)
        prior = self.distribution(**inputs)
        self._pymc3_priors[self.label(i)] = prior
        return prior

    def get_prior(self, i, *args, **kwargs):
        """
        Get the PyMC3 prior for this wavelength.

        Parameters
        ----------
        i : int
            The index of the wavelength associated with this prior.

        Returns
        -------
        prior : PyMC3 distribution
            The prior for this parameter for this wavelength
        """
        try:
            return self._pymc3_priors[self.label(i)]
        except KeyError:
            return self.generate_pymc3(i)

    def generate_pymc3_vector(self, shape, i=None, *args, **kwargs):
        """
        Generate a PyMC3 prior for wavelength i.

        Parameters
        ----------
        shape : int
            The number of wavelengths associated with this prior.
        kw : dict
            All keyword arguments will be ignored.
        """
        inputs = self.inputs.copy()
        # if a different prior is passed for every wavelength then we use only the one for wavelength i
        # for k, v in self.inputs.items():
        #     if len(np.shape(v)) > 1:
        #         inputs[k] = v[i]

        inputs = dict(**inputs)
        if "shape" not in self.inputs:
            inputs["shape"] = (shape, 1)
        else:
            if type(self.inputs["shape"]) == int:
                inputs["shape"] = (shape, self.inputs["shape"])
                if "testval" in inputs:
                    inputs["testval"] = [inputs["testval"]] * shape
            else:
                if len(self.inputs["shape"]) == 1:
                    inputs["shape"] = (shape, self.inputs["shape"])
                    if "testval" in inputs:
                        inputs["testval"] = inputs["testval"] * shape

        self.inputs["shape"] = inputs["shape"]
        prior = self.distribution(**inputs)
        self._pymc3_prior = prior
        return prior

    def get_prior_vector(self, shape=1, *args, **kwargs):
        """
        Get the PyMC3 prior for this wavelength.

        Parameters
        ----------
        shape : int
            The number of wavelengths associated with this prior.

        Returns
        -------
        prior : PyMC3 distribution
            The prior for this parameter
        """
        try:
            return self._pymc3_prior
        except AttributeError:
            return self.generate_pymc3_vector(shape)

    def clear_prior(self, *args, **kwargs):
        """
        Clear the stored PyMC3 prior.
        """
        try:
            delattr(self, "_pymc3_priors")
            delattr(self, "_pymc3_prior")
            print(f"Cleared {self.name} prior")
            self._pymc3_priors = {}
        except AttributeError:
            pass

    def __repr__(self):
        distribution_name = self.distribution.__name__
        inputs_as_string = ", ".join([f"{k}={repr(v)}" for k, v in self.inputs.items()])
        return f"<🧮 WavelikeFitted {distribution_name}({inputs_as_string}) for each wavelength 🧮>"
