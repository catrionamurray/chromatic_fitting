# from chromatic_fitting.archive.chromatic_fitting import *
from chromatic_fitting.imports import *

# from chromatic_fitting.spectrum import *
from .models import *
from .utils import *
from tqdm import tqdm
from .parameters import *
from arviz import summary
from pymc3_ext import eval_in_model, optimize, sample
from pymc3 import (
    sample_prior_predictive,
    sample_posterior_predictive,
    Deterministic,
    Normal,
    TruncatedNormal,
)
import warnings
import collections

#
# #
# def add_dicts(dict_1: dict, dict_2: dict) -> dict:
#     """
#     Combine two dictionaries, if they have the same key then add the values of that key
#
#     Parameters
#     ----------
#     dict_1: first dictionary to add
#     dict_2: second dictionary to add
#
#     Returns
#     -------
#     object
#
#     """
#     # combine the keys that are unique to dict_1 or dict_2 into new dict_3 (order doesn't matter for addition)
#     dict_3 = {**dict_2, **dict_1}
#     # for the keys that appear in both add the values
#     for key, value in dict_3.items():
#         if key in dict_1 and key in dict_2:
#             dict_3[key] = np.array(value) + np.array(dict_2[key])
#     return dict_3
#
#
# def subtract_dicts(dict_1: dict, dict_2: dict) -> dict:
#     """
#     Combine two dictionaries, if they have the same key then subtract the values of second dictionary from first
#
#     Parameters
#     ----------
#     dict_1: first dictionary
#     dict_2: second dictionary to subtract from first
#
#     Returns
#     -------
#     object
#
#     """
#     # # combine the keys that are unique to dict_1 or dict_2 into new dict_3 (order matters for subtraction!)
#     dict_3 = {**dict_2, **dict_1}
#     # for the keys that appear in both subtract values in dict_2 from value in dict_1
#     for key, value in dict_3.items():
#         if key in dict_1 and key in dict_2:
#             dict_3[key] = np.array(value) - np.array(dict_2[key])
#     return dict_3
#
#
# def multiply_dicts(dict_1: dict, dict_2: dict) -> dict:
#     """
#     Combine two dictionaries, if they have the same key then multiply the values of second dictionary by first
#
#     Parameters
#     ----------
#     dict_1: first dictionary
#     dict_2: second dictionary
#
#     Returns
#     -------
#     object
#
#     """
#     # combine the keys that are unique to dict_1 or dict_2 into new dict_3 (order doesn't matter for multiplication)
#     dict_3 = {**dict_2, **dict_1}
#     # for the keys that appear in both multiply values in dict_2 by value in dict_1
#     for key, value in dict_3.items():
#         if key in dict_1 and key in dict_2:
#             dict_3[key] = np.array(value) * np.array(dict_2[key])
#     return dict_3
#
#
# def divide_dicts(dict_1: dict, dict_2: dict) -> dict:
#     """
#     Combine two dictionaries, if they have the same key then divide the values of first dictionary by second
#
#     Parameters
#     ----------
#     dict_1: first dictionary to divide by second
#     dict_2: second dictionary
#
#     Returns
#     -------
#     object
#
#     """
#     # combine the keys that are unique to dict_1 or dict_2 into new dict_3 (order matters for division!)
#     dict_3 = {**dict_2, **dict_1}
#     # for the keys that appear in both divide values in dict_1 by value in dict_2
#     for key, value in dict_3.items():
#         if key in dict_1 and key in dict_2:
#             dict_3[key] = np.array(value) / np.array(dict_2[key])
#     return dict_3
#
#
# # define a dictionary mapping operations to functions
# combination_options = {
#     "+": add_dicts,
#     "-": subtract_dicts,
#     "*": multiply_dicts,
#     "/": divide_dicts,
# }
#
#
# class LightcurveModel:
#     required_parameters = []
#
#     def __init__(self, name="lightcurve_model", **kw):
#         # define some default parameters (fixed):
#         self.defaults = dict()
#         self.optimization = "simultaneous"
#         self.store_models = False
#         self.name = name
#         self.outlier_flag = False
#         # pass
#
#     def __add__(self, other):
#         """
#         What should we return for `a + b` for two models `a` and `b`?
#         """
#         cm = CombinedModel()
#         cm.initialize_empty_model()
#         cm.combine(self, other, "+")
#
#         return cm
#
#     def __sub__(self, other):
#         """
#         What should we return for `a - b` for two models `a` and `b`?
#         """
#         cm = CombinedModel()
#         cm.initialize_empty_model()
#         cm.combine(self, other, "-")
#
#         return cm
#
#     def __mul__(self, other):
#         """
#         What should we return for `a * b` for two models `a` and `b`?
#         """
#         cm = CombinedModel()
#         cm.initialize_empty_model()
#         cm.combine(self, other, "*")
#
#         return cm
#
#     def __truediv__(self, other):
#         """
#         What should we return for `a - b` for two models `a` and `b`?
#         """
#         cm = CombinedModel()
#         cm.initialize_empty_model()
#         cm.combine(self, other, "/")
#         return cm
#
#     def setup_parameters(self, **kw):
#         """
#         Set the values of the model parameters.
#
#         Parameters
#         ----------
#         kw : dict
#             All keyword arguments will be treated as model
#             parameters and stored in the `self.parameters`
#             dictionary attribute. This input both sets
#             the initial values for each parameter and
#             indicates whether it should be fixed or fitted,
#             and if it is fitted what its prior should be.
#         """
#
#         # set up a dictionary of unprocessed
#         # print("RUNNING SETUP PARAMS")
#         # print(self.defaults)
#         unprocessed_parameters = dict(self.defaults)
#         unprocessed_parameters.update(**kw)
#         # print(unprocessed_parameters)
#
#         # process all parameters into Parameter objects
#         self.parameters = {}
#         for k, v in unprocessed_parameters.items():
#             if isinstance(v, Parameter):
#                 self.parameters[k] = v
#                 self.parameters[k].set_name(k)
#             elif isinstance(v, LightcurveModel):
#                 # if a LightcurveModel is passed as a prior for a variable then copy it, but
#                 # do not run the setup_lightcurves (weird pymc3 inheritance issues) - this will be called from within
#                 # the main setup_lightcurves!
#                 if hasattr(v, "every_light_curve"):
#                     new_v = v.__class__()
#                     new_v.initialize_empty_model()
#                     # can't just do new_v.setup_parameters(**dict(v.parameters)) - runs into weird pymc3 inheritance
#                     # issues!
#                     new_params = {}
#                     for k2, v2 in v.parameters.items():
#                         new_params[k2] = v2.__class__(v2.distribution, **v2.inputs)
#                     new_v.setup_parameters(**new_params)
#                     # if I've already attached data to this model then attach it to the new model:
#                     if hasattr(v, "data"):
#                         new_v.attach_data(v.data)
#                 else:
#                     new_v = v
#                 self.parameters[k] = new_v
#                 self.parameters[k].set_name(new_v)
#             else:
#                 self.parameters[k] = Fixed(v)
#                 self.parameters[k].set_name(v)
#
#         # check that all the necessary parameters are defined somehow
#         for k in self.required_parameters:
#             assert k in self.parameters
#
#         for k in self.parameters.keys():
#             if self.name in k:
#                 return
#         self.parameters = add_string_before_each_dictionary_key(
#             self.parameters, self.name
#         )
#
#         for v in self.parameters.values():
#             v.set_name(f"{self.name}_{v.name}")
#
#     def get_parameter_shape(self, param):
#         inputs = param.inputs
#         for k in ["testval", "mu", "lower", "upper", "sigma"]:
#             if k in inputs.keys():
#                 if (type(inputs[k]) == int) or (type(inputs[k]) == float):
#                     inputs["shape"] = 1
#                 elif len(inputs[k]) == 1:
#                     inputs["shape"] = 1
#                 else:
#                     inputs["shape"] = len(inputs[k])
#                 break
#         param.inputs = inputs
#
#     def summarize_parameters(self):
#         """
#         Print a friendly summary of the parameters.
#         """
#         for k, v in self.parameters.items():
#             print(f"{k} =\n  {v}\n")
#
#     def reinitialize_parameters(self, exclude=[]):
#         """
#         Remove the pymc3 prior model from every parameter not in exclude
#         """
#         for k, v in self.parameters.items():
#             if k not in exclude:
#                 if isinstance(v, Fitted):
#                     v.clear_prior()
#
#     def extract_extra_class_inputs(self):
#         """
#         Extract any additional keywords passed to the LightcurveModel
#         """
#         class_inputs = {}
#         varnames_to_remove = ["defaults", "optimization", "pymc3_model", "parameters"]
#         for k, v in self.__dict__.items():
#             if k not in varnames_to_remove:
#                 class_inputs[k] = v
#         return class_inputs
#
#     def initialize_empty_model(self):
#         """
#         Restart with an empty model.
#         """
#         self.pymc3_model = pm.Model()
#
#     def attach_data(self, rainbow):
#         """
#         Connect a `chromatic` Rainbow dataset to this object.
#         """
#         self.data = rainbow._create_copy()
#
#     def white_light_curve(self):
#         """
#         Generate inverse-variance weighted white light curve by binning Rainbow to one bin
#         """
#         # if self.outlier_flag:
#         #     data = self.data_without_outliers
#         # else:
#         #     data = self.data
#         self.white_light = self.data.bin(nwavelengths=self.data.nwave)
#
#     def choose_optimization_method(self, optimization_method="simultaneous"):
#         """
#         Choose the optimization method
#         [attach_data has to be run before!]
#         """
#         possible_optimization_methods = ["simultaneous", "white_light", "separate"]
#         if optimization_method in possible_optimization_methods:
#             self.optimization = optimization_method
#             if self.optimization == "separate":
#                 try:
#                     self.create_multiple_models()
#                     self.change_all_priors_to_Wavelike()
#                 except:
#                     pass
#         else:
#             print(
#                 "Unrecognised optimization method, please select one of: "
#                 + str(", ".join(possible_optimization_methods))
#             )
#             self.optimization = "simultaneous"
#
#     def create_multiple_models(self):
#         """
#         Create a list of models to process wavelengths separately
#         """
#         self.pymc3_model = [pm.Model() for n in range(self.data.nwave)]
#         # if the LightCurve model is a CombinedModel then update the constituent models too
#         if isinstance(self, CombinedModel):
#             for mod in self.chromatic_models.values():
#                 mod.pymc3_model = self.pymc3_model
#                 mod.optimization = self.optimization
#
#     def change_all_priors_to_Wavelike(self):
#         for k, v in self.parameters.items():
#             if isinstance(v, Fitted) and not isinstance(v, WavelikeFitted):
#                 self.parameters[k] = WavelikeFitted(v.distribution, **v.inputs)
#             if isinstance(v, Fixed) and not isinstance(v, WavelikeFixed):
#                 self.parameters[k] = WavelikeFixed([v.value] * self.data.nwave)
#
#     def change_all_priors_to_notWavelike(self):
#         for k, v in self.parameters.items():
#             if isinstance(v, WavelikeFitted):
#                 self.parameters[k] = Fitted(v.distribution, **v.inputs)
#             if isinstance(v, WavelikeFixed):
#                 self.parameters[k] = Fixed(v.values[0])
#
#     def separate_wavelengths(self, i):
#         # if self.outlier_flag:
#         #     data_copy = self.data_without_outliers._create_copy()
#         # else:
#         data_copy = self.data._create_copy()
#
#         for k, v in data_copy.fluxlike.items():
#             data_copy.fluxlike[k] = np.array([data_copy.fluxlike[k][i, :]])
#         for k, v in data_copy.wavelike.items():
#             data_copy.wavelike[k] = [data_copy.wavelike[k][i]]
#         return data_copy
#
#     def get_data(self, i=0):
#         """
#         Extract the data to use for the optimization depending on the method chosen
#         """
#         if hasattr(self, "optimization"):
#             if self.optimization == "white_light":
#                 if hasattr(self, "white_light"):
#                     return self.white_light
#                 self.white_light_curve()
#                 return self.white_light
#             if self.optimization == "separate":
#                 return self.separate_wavelengths(i)
#         # if self.outlier_flag:
#         #     return self.data_without_outliers
#         # else:
#         return self.data
#
#     def setup_likelihood(
#         self,
#         mask_outliers=False,
#         mask_wavelength_outliers=False,
#         sigma_wavelength=5,
#         data_mask=None,
#         inflate_uncertainties=False,
#         **kw,
#     ):
#         """
#         Connect the light curve model to the actual data it aims to explain.
#         """
#         # data = self.get_data()
#         self.bad_wavelengths = []
#
#         if self.optimization == "separate":
#             models = self.pymc3_model
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#             data = self.data
#         else:
#             models = [self.pymc3_model]
#             data = self.get_data()
#             datas = [data]
#
#         # if the data has outliers, then mask them out
#         if mask_outliers:
#             # if the user has specified a mask, then use that
#             if data_mask is None:
#                 # sigma-clip in time
#                 data_mask = np.array(get_data_outlier_mask(data, **kw))
#                 if mask_wavelength_outliers:
#                     # sigma-clip in wavelength
#                     data_mask[
#                         get_data_outlier_mask(
#                             data, clip_axis="wavelength", sigma=sigma_wavelength
#                         )
#                         == True
#                     ] = True
#                 # data_mask_wave =  get_data_outlier_mask(data, clip_axis='wavelength', sigma=4.5)
#             self.outlier_mask = data_mask
#             self.outlier_flag = True
#             self.data_without_outliers = remove_data_outliers(data, data_mask)
#
#         if inflate_uncertainties:
#             self.parameters["nsigma"] = WavelikeFitted(
#                 TruncatedNormal, mu=1, sd=0.1, lower=1
#             )
#             self.parameters["nsigma"].set_name("nsigma")
#
#         for j, (mod, data) in enumerate(zip(models, datas)):
#             with mod:
#                 for i, w in enumerate(data.wavelength):
#                     k = f"wavelength_{j + i}"
#
#                     if inflate_uncertainties:
#                         uncertainties = data.uncertainty[i, :] * eval_in_model(
#                             self.parameters["nsigma"].get_prior(j + i)
#                         )
#                     else:
#                         uncertainties = data.uncertainty[i, :]
#
#                     try:
#                         # if the user has passed mask_outliers=True then sigma clip and use the outlier mask
#                         if mask_outliers:
#                             flux = self.data_without_outliers.flux[i + j, :]
#                         else:
#                             flux = data.flux[i, :]
#
#                         pm.Normal(
#                             f"{k}_data",
#                             mu=self.every_light_curve[k],
#                             sd=uncertainties,
#                             observed=flux,
#                         )
#                     except Exception as e:
#                         print(f"Setting up likelihood failed for wavelength {i}: {e}")
#                         self.bad_wavelengths.append(i)
#
#     def sample_prior(self, ndraws=3):
#         """
#         Draw samples from the prior distribution.
#         :parameter n
#         Number of priors to sample
#         """
#         try:
#             with self.pymc3_model:
#                 return sample_prior_predictive(ndraws)
#         except:
#             priors = []
#             for mod in self.pymc3_model:
#                 with mod:
#                     priors.append(sample_prior_predictive(ndraws))
#             return priors
#
#     def sample_posterior(self, ndraws=3, var_names=None):
#         """
#         Draw samples from the posterior distribution.
#         :parameter n
#         Number of posteriors to sample
#         """
#         if not hasattr(self, "trace"):
#             print("Sampling has not been run yet! Running now with defaults...")
#             self.sample()
#
#         if self.optimization != "separate":
#             with self.pymc3_model:
#                 return sample_posterior_predictive(
#                     self.trace, ndraws, var_names=var_names
#                 )
#         else:
#             posteriors = []
#             for mod, trace in zip(self.pymc3_model, self.trace):
#                 with mod:
#                     posteriors.append(
#                         sample_posterior_predictive(trace, ndraws, var_names=var_names)
#                     )
#             return posteriors
#
#     def optimize(self, plot=True, plotkw={}, **kw):
#         """
#         Wrapper for PyMC3_ext sample
#         """
#         if self.optimization == "separate":
#             opts = []
#             if "start" in kw:
#                 start = kw["start"]
#                 kw.pop("start")
#                 for mod, opt in zip(self.pymc3_model, start):
#                     with mod:
#                         opts.append(optimize(start=opt, **kw))
#             else:
#                 for mod in self.pymc3_model:
#                     with mod:
#                         opts.append(optimize(**kw))
#             if plot:
#                 self.plot_optimization(opts, **plotkw)
#             return opts
#         else:
#             with self.pymc3_model:
#                 opt = optimize(**kw)
#             if plot:
#                 self.plot_optimization(opt, **plotkw)
#             return opt
#
#     def plot_optimization(self, opt, offset=0.03, figsize=(6, 18)):
#         if self.optimization == "separate":
#             opts = opt
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#         else:
#             opts = [opt]
#             datas = [self.get_data()]
#
#         plt.figure(figsize=figsize)
#         for j, (opt_sep, data) in enumerate(zip(opts, datas)):
#             for w in range(data.nwave):
#                 try:
#                     if w == 0:
#                         plt.plot(
#                             data.time,
#                             ((w + j) * offset) + opt_sep[f"{self.name}_model_w{w + j}"],
#                             "k",
#                             label=self.name,
#                         )
#                     else:
#                         plt.plot(
#                             data.time,
#                             ((w + j) * offset) + opt_sep[f"{self.name}_model_w{w + j}"],
#                             "k",
#                         )
#
#                     if isinstance(self, CombinedModel):
#                         if w == 0:
#                             for mod in self.chromatic_models.values():
#                                 plt.plot(
#                                     data.time,
#                                     ((w + j) * offset)
#                                     + opt_sep[f"{mod.name}_model_w{w + j}"],
#                                     label=mod.name,
#                                 )
#                         else:
#                             for mod in self.chromatic_models.values():
#                                 plt.plot(
#                                     data.time,
#                                     ((w + j) * offset)
#                                     + opt_sep[f"{mod.name}_model_w{w + j}"],
#                                 )
#                 except:
#                     pass
#                 plt.plot(data.time, ((w + j) * offset) + data.flux[w, :], "k.")
#                 plt.errorbar(
#                     data.time,
#                     ((w + j) * offset) + data.flux[w, :],
#                     data.uncertainty[w, :],
#                     color="k",
#                     linestyle="None",
#                     capsize=2,
#                 )
#         plt.legend()
#         plt.show()
#
#     def sample_individual(self, i, **kw):
#         """
#         Wrapper for PyMC3_ext sample - only for a single wavelength i
#         """
#         starts = []
#         if "start" in kw:
#             starts = kw["start"]
#             kw.pop("start")
#
#         with self.pymc3_model[i] as mod:
#             if len(starts) > 0:
#                 start = starts[i]
#             else:
#                 start = mod.test_point
#
#             try:
#                 return sample(start=start, **kw)
#             except Exception as e:
#                 print(f"Sampling failed for one of the models: {e}")
#                 return None
#
#     def sample(self, summarize_step_by_step=False, summarize_kw={}, **kw):
#         """
#         Wrapper for PyMC3_ext sample
#         """
#         if self.optimization == "separate":
#             self.trace = []
#             starts = []
#             if "start" in kw:
#                 starts = kw["start"]
#                 kw.pop("start")
#
#             for i, mod in enumerate(self.pymc3_model):
#                 with mod:
#                     if len(starts) > 0:
#                         start = starts[i]
#                     else:
#                         start = mod.test_point
#
#                     try:
#                         samp = sample(start=start, **kw)
#                         if summarize_step_by_step:
#                             self.summary.append(summary(samp, **summarize_kw))
#                         else:
#                             self.trace.append(samp)
#
#                     except Exception as e:
#                         print(f"Sampling failed for one of the models: {e}")
#                         self.trace.append(None)
#
#             if summarize_step_by_step:
#                 if isinstance(self, CombinedModel):
#                     for m in self.chromatic_models.values():
#                         m.summary = self.summary
#
#             # for mod in self.pymc3_model:
#             #     with mod:
#             #         try:
#             #             self.trace.append(sample(**kw))
#             #         except Exception as e:
#             #             print(f"Sampling failed for one of the models!: {e}")
#             #             self.trace.append(None)
#         else:
#             with self.pymc3_model:
#                 self.trace = sample(**kw)
#
#     def summarize(self, print_table=True, **kw):
#         """
#         Wrapper for arviz summary
#         """
#         if not hasattr(self, "trace"):
#             print("Sampling has not been run yet! Running now with defaults...")
#             self.sample()
#
#         if hasattr(self, "summary"):
#             print("Summarize has already been run")
#             if print_table:
#                 print(self.summary)
#             return
#
#         if self.optimization == "separate":
#             self.summary = []
#             for mod, trace in zip(self.pymc3_model, self.trace):
#                 with mod:
#                     self.summary.append(summary(trace, **kw))
#         else:
#             with self.pymc3_model:
#                 self.summary = summary(self.trace, **kw)
#
#         if print_table:
#             print(self.summary)
#
#         if isinstance(self, CombinedModel):
#             for m in self.chromatic_models.values():
#                 m.summary = self.summary
#
#     def get_results(self, as_df=True, uncertainty=["hdi_3%", "hdi_97%"]):
#         """
#         Extract mean results from summary
#         """
#
#         # if the user wants to have the same uncertainty for lower and upper error:
#         if type(uncertainty) == str:
#             uncertainty = [uncertainty, uncertainty]
#
#         results = {}
#         for i, w in enumerate(self.data.wavelength):
#             transit_params_mean = self.extract_from_posteriors(self.summary, i)
#             transit_params_lower_error = self.extract_from_posteriors(
#                 self.summary, i, op=uncertainty[0]
#             )
#             transit_params_lower_error = dict(
#                 (key + f"_{uncertainty[0]}", value)
#                 for (key, value) in transit_params_lower_error.items()
#             )
#             transit_params_upper_error = self.extract_from_posteriors(
#                 self.summary, i, op=uncertainty[1]
#             )
#             transit_params_upper_error = dict(
#                 (key + f"_{uncertainty[1]}", value)
#                 for (key, value) in transit_params_upper_error.items()
#             )
#             transit_params = (
#                 transit_params_mean
#                 | transit_params_lower_error
#                 | transit_params_upper_error
#             )
#             ordered_transit_params = collections.OrderedDict(
#                 sorted(transit_params.items())
#             )
#             results[f"w{i}"] = ordered_transit_params
#             results[f"w{i}"]["wavelength"] = w
#
#         if as_df:
#             # if the user wants to return the results as a pandas DataFrame:
#             return pd.DataFrame(results).transpose()
#         else:
#             # otherwise return a dictionary of dictionaries
#             return results
#
#     def make_transmission_spectrum_table(
#         self, uncertainty=["hdi_3%", "hdi_97%"], svname=None
#     ):
#         """
#         Generate and return a transmission spectrum table
#         """
#         results = self.get_results(uncertainty=uncertainty)[
#             [
#                 "wavelength",
#                 f"{self.name}_radius_ratio",
#                 f"{self.name}_radius_ratio_{uncertainty[0]}",
#                 f"{self.name}_radius_ratio_{uncertainty[1]}",
#             ]
#         ]
#         if svname is not None:
#             results.to_csv(svname)
#         else:
#             return results
#
#     def run_simultaneous_fit(self, r, **kwargs):
#         """
#         Run the entire simultaneous wavelength fit.
#         """
#         self.attach_data(r)
#         self.setup_lightcurves()
#         self.setup_likelihood()
#         opt = self.optimize(start=self.pymc3_model.test_point)
#         opt = self.optimize(start=opt)
#         self.sample(start=opt)
#         self.summarize(round_to=7, fmt="wide")
#
#     # def remove_data_outliers(self, **kw):
#     #     """
#     #     Remove outliers from the data.
#     #     [Ideally to be replaced with some chromatic.flag_outliers() function]
#     #     """
#     #     data_outliers_removed = remove_data_outliers(self.data, **kw)
#     #     self.data_outliers_removed = data_outliers_removed
#     #     self.outlier_flag = True
#
#     def plot_priors(self, n=3):
#         """
#         Plot n prior samples from the parameter distributions defined by the user
#         :parameter n
#         Number of priors to plot (default=3)
#         """
#         # setup the models, data and orbits in a format for looping
#         if self.optimization == "separate":
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#             prior_predictive_traces = self.sample_prior(ndraws=n)
#         else:
#             datas = [self.get_data()]
#             prior_predictive_traces = [self.sample_prior(ndraws=n)]
#
#         for nm, (data, prior_predictive_trace) in enumerate(
#             zip(datas, prior_predictive_traces)
#         ):
#             for i in range(n):
#                 flux_for_this_sample = np.array(
#                     [
#                         prior_predictive_trace[f"wavelength_{w + nm}_data"][i]
#                         for w in range(data.nwave)
#                     ]
#                 )
#                 # model_for_this_sample = np.array(
#                 #     [
#                 #         prior_predictive_trace[f"{self.name}_model_w{w + nm}"][i]
#                 #         for w in range(data.nwave)
#                 #     ]
#                 # )
#                 # f"{name}model_w{i + j}"
#                 data.fluxlike[f"prior-predictive-{i}"] = flux_for_this_sample
#                 # data.fluxlike[f"prior-model-{i}"] = model_for_this_sample
#             data.imshow_quantities()
#
#     def plot_posteriors(self, n=3):
#         """
#         Plot n posterior samples from the parameter distributions defined by the user
#         :parameter trace
#         PyMC3 trace object (need to run sample first)
#         :parameter n
#         Number of posteriors to plot (default=3)
#         """
#         # if we have the separate wavelength optimization method chosen then repeat for every wavelength/model
#         if self.optimization == "separate":
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#             posterior_predictive_traces = self.sample_posterior(n)
#         else:
#             datas = [self.get_data()]
#             posterior_predictive_traces = [self.sample_posterior(n)]
#
#         posterior_model = {}
#         for nm, (data, posterior_predictive_trace) in enumerate(
#             zip(datas, posterior_predictive_traces)
#         ):
#
#             for w in range(data.nwave):
#                 if f"wavelength_{w + nm}_data" in posterior_predictive_trace.keys():
#                     # generate a posterior model for every wavelength:
#                     posterior_model[
#                         f"{self.name}_model_w{w + nm}"
#                     ] = self.sample_posterior(
#                         n, var_names=[f"{self.name}_model_w{w + nm}"]
#                     )[
#                         f"{self.name}_model_w{w + nm}"
#                     ]
#
#             for i in range(n):
#                 # for every posterior sample extract the posterior model and distribution draw for every wavelength:
#                 flux_for_this_sample = np.array(
#                     [
#                         posterior_predictive_trace[f"wavelength_{w + nm}_data"][i]
#                         for w in range(data.nwave)
#                     ]
#                 )
#                 model_for_this_sample = np.array(
#                     [
#                         posterior_model[f"{self.name}_model_w{w + nm}"][i]
#                         for w in range(data.nwave)
#                     ]
#                 )
#
#                 # add posterior model and draw from posterior distribution to the Rainbow quantities:
#                 data.fluxlike[f"posterior-model-{i}"] = model_for_this_sample
#                 data.fluxlike[f"posterior-predictive-{i}"] = flux_for_this_sample
#             # plot the rainbow quantities:
#             data.imshow_quantities()
#
#     def extract_deterministic_model(self):
#         """
#         Extract the deterministic model from the summary statistics
#         """
#         if self.optimization == "separate":
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#         else:
#             datas = [self.get_data()]
#
#         model = {}
#         for nm, data in enumerate(datas):
#             for w in range(data.nwave):
#                 if self.optimization == "separate":
#                     summary = self.summary[w + nm]
#                 else:
#                     summary = self.summary
#
#                 if f"w{w + nm}" not in model.keys():
#                     model[f"w{w + nm}"] = []
#                 for t in range(data.ntime):
#                     model[f"w{w + nm}"].append(
#                         summary["mean"][f"{self.name}_model_w{w + nm}[{t}]"]
#                     )
#         return model
#
#     def get_model(self, as_dict=True, as_array=False):
#         """
#         Return the 'best-fit' model from the summary table as a dictionary or as an array
#         """
#         model = self.extract_deterministic_model()
#
#         if as_array:
#             return np.array(list(model.values()))
#         elif as_dict:
#             return model
#
#     def plot_lightcurves(self, detrend=False, t_unit="day", ax=None, **kw):
#         if not hasattr(self, "summary"):
#             print(
#                 "The summarize step has not been run yet. To include the 'best-fit' model please run "
#                 "{self}.summarize() before calling this step!"
#             )
#             add_model = False
#         else:
#             if isinstance(self, CombinedModel):
#                 model = self.get_model()
#             else:
#                 model = {"total": self.get_model()}
#             add_model = True
#
#         if ax is None:
#             ax = plt.subplot()
#         plt.sca(ax)
#
#         # if self.optimization == "separate":
#         #     datas = [self.get_data(i) for i in range(self.data.nwave)]
#         # else:
#         #     datas = [self.get_data()]
#         if self.optimization != "separate":
#             data = self.get_data()
#         else:
#             data = self.data
#
#         if self.outlier_flag:
#             data.plot(ax=ax, cmap="Reds", **kw)
#             self.data_without_outliers.plot(ax=ax, **kw)
#         else:
#             data.plot(ax=ax, **kw)
#
#         # for data in datas:
#         spacing = ax._most_recent_chromatic_plot_spacing
#
#         if add_model:
#             for i in range(data.nwave):
#                 ax.plot(
#                     data.time.to_value(t_unit),
#                     model["total"][f"w{i}"] - (i * spacing),
#                     color="k",
#                 )
#         plt.show()
#         plt.close()
#
#         # if add_model and detrend:
#
#     def extract_from_posteriors(self, summary, i, op="mean"):
#         # there"s definitely a sleeker way to do this
#         if self.optimization == "separate":
#             summary = summary[i]
#
#         # ensure that the specified operation is in the summary table!
#         if op not in summary:
#             print(
#                 f"Try a different operation or regenerate the summary table! "
#                 f"{op} is not currently in the summary table"
#             )
#             return None
#
#         posterior_means = summary[op]
#         fv = {}
#         for k, v in self.parameters.items():
#             if k in posterior_means.index:
#                 fv[k] = posterior_means[k]
#             elif f"{k}[0]" in posterior_means.index:
#                 n = 0
#                 fv[k] = []
#                 while f"{k}[{n}]" in posterior_means.index:
#                     fv[k].append(posterior_means[f"{k}[{n}]"])
#                     n += 1
#             elif f"{k}_w{i}" in posterior_means.index:
#                 fv[k] = posterior_means[f"{k}_w{i}"]
#             elif f"{k}_w{i}[0]" in posterior_means.index:
#                 n = 0
#                 fv[k] = []
#                 while f"{k}_w{i}[{n}]" in posterior_means.index:
#                     fv[k].append(posterior_means[f"{k}_w{i}[{n}]"])
#                     n += 1
#             elif f"{k}_w{i}" in posterior_means.index:
#                 fv[k] = posterior_means[f"{k}_w{i}"]
#             else:
#                 if isinstance(v, WavelikeFixed):
#                     fv[k] = v.values[0]
#                 elif isinstance(v, Fixed):
#                     fv[k] = v.value
#
#         return fv
#
#     def imshow_with_models(self, **kw):
#         """
#         Plot the lightcurves with the best-fit models (if they exist).
#         [Wrapper for chromatic]
#         """
#         if not hasattr(self, "data_with_model"):
#             print(
#                 "No model attached to data. Running `add_model_to_rainbow` now. You can access this data later"
#                 " using [self].data_with_model"
#             )
#             self.add_model_to_rainbow()
#         self.data_with_model.imshow_with_models(**kw)
#
#     def animate_with_models(self, **kw):
#         """
#         Animate the lightcurves with the best-fit models (if they exist).
#         [Wrapper for chromatic]
#         """
#         if not hasattr(self, "data_with_model"):
#             print(
#                 "No model attached to data. Running `add_model_to_rainbow` now. You can access this data later"
#                 "using [self].data_with_model"
#             )
#             self.add_model_to_rainbow()
#         self.data_with_model.animate_with_models(**kw)
#
#     def plot_with_model_and_residuals(self, **kw):
#         """
#         Plot the lightcurves with the best-fit models (if they exist) and residuals.
#         [Wrapper for chromatic]
#         """
#         if not hasattr(self, "data_with_model"):
#             print(
#                 "No model attached to data. Running `add_model_to_rainbow` now. You can access this data later"
#                 "using [self].data_with_model"
#             )
#             self.add_model_to_rainbow()
#         self.data_with_model.plot_with_model_and_residuals(**kw)
#
#
# class CombinedModel(LightcurveModel):
#     def __init__(self, name="combined", **kw):
#         super().__init__(name, **kw)
#         self.name = name
#         self.metadata = {}
#         self.parameters = {}
#
#     def __repr__(self):
#         if hasattr(self, "chromatic_models"):
#             string_to_print = f"<chromatic combined model 🌈, models: "
#             for i, (model_name, model) in enumerate(self.chromatic_models.items()):
#                 if i >= len(self.how_to_combine):
#                     string_to_print += f"{model_name}({model})"
#                 else:
#                     string_to_print += (
#                         f"{model_name}({model}) {self.how_to_combine[i]} "
#                     )
#             return string_to_print
#             # return f"({self.chromatic_models['left']} {self.how_to_combine} {self.chromatic_models['right']})"
#             # return f"<experimental chromatic combined model 🌈, models: {self.chromatic_models}>"
#         else:
#             return "<chromatic combined model 🌈>"
#
#     def combine(self, first, second, how_to_combine):
#         if isinstance(first, CombinedModel) and isinstance(second, CombinedModel):
#             # if both first and second are CombinedModels
#             chromatic_models = add_dicts(
#                 first.chromatic_models.copy(), second.chromatic_models.copy()
#             )
#             self.how_to_combine = first.how_to_combine + second.how_to_combine
#             self.attach_models(chromatic_models, how_to_combine=how_to_combine)
#         elif isinstance(first, CombinedModel):
#             # if the first is a CombinedModel
#             chromatic_models = first.chromatic_models.copy()
#             chromatic_models[f"{second.name}"] = second
#             self.how_to_combine = first.how_to_combine
#             self.attach_models(chromatic_models, how_to_combine=how_to_combine)
#         elif isinstance(second, CombinedModel):
#             # if the second is a CombinedModel
#             chromatic_models = second.chromatic_models.copy()
#             chromatic_models[f"{first.name}"] = first
#             self.how_to_combine = second.how_to_combine
#             self.attach_models(chromatic_models, how_to_combine=how_to_combine)
#         else:
#             # if neither is a CombinedModel
#             self.attach_models(
#                 {f"{first.name}": first, f"{second.name}": second},
#                 how_to_combine=how_to_combine,
#             )
#
#     def attach_models(self, models, how_to_combine="+"):
#         """
#         Attach multiple LightCurveModel in dictionary to the CombinedModel
#         """
#         new_models = {}
#
#         # if we have already attached models with instructions on how to combine then add this operation
#         # to self.how_to_combine
#         if hasattr(self, "how_to_combine"):
#             self.how_to_combine.append(how_to_combine)
#         else:
#             if type(how_to_combine) == str:
#                 # if a string operation ("+","-", etc.) has been passed then repeat this operation for every model
#                 self.how_to_combine = [how_to_combine] * (len(models.keys()) - 1)
#             else:
#                 # if we have passed a list of operations of the same length as (n_models-1) then save
#                 if len(how_to_combine) == len(models.keys()) - 1:
#                     self.how_to_combine = how_to_combine
#                 else:
#                     print(
#                         f"WARNING: You have passed {len(how_to_combine)} operations for {len(models.keys())} models!"
#                     )
#
#         for name, model in models.items():
#             # check that the models passed to this function are LightcurveModels
#             if isinstance(model, LightcurveModel):
#                 # make a "copy" of each model:
#                 class_inputs = model.extract_extra_class_inputs()
#                 new_model = model.__class__(**class_inputs)
#                 new_model.pymc3_model = self.pymc3_model
#                 # can't just do new_v.setup_parameters(**dict(v.parameters)) - runs into weird pymc3 inheritance
#                 # issues! Probably because of hidden __prior__ saved
#                 model_params = {}
#                 # for every parameter in the separate models redefine them in the separate and new models
#                 for k, v in model.parameters.items():
#                     # print(name, k)
#                     if isinstance(v, WavelikeFixed):
#                         # parameter is WavelikeFixed
#                         model_params[k] = v.__class__(v.values)
#                     elif isinstance(v, Fixed):
#                         # parameter is Fixed
#                         model_params[k] = v.__class__(v.value)
#                     else:
#                         # parameter is Fitted or WavelikeFitted
#                         model_params[k] = v.__class__(v.distribution, **v.inputs)
#                 # set up parameters in new models
#                 new_model.defaults = add_string_before_each_dictionary_key(
#                     new_model.defaults, new_model.name
#                 )
#                 new_model.required_parameters = [
#                     f"{new_model.name}_{a}" for a in new_model.required_parameters
#                 ]
#                 new_model.setup_parameters(**model_params)
#                 new_models[name] = new_model
#                 # all_params = all_params | model_params # what happens if same keys?
#             else:
#                 print("This class can only be used to combine LightcurveModels!")
#
#         # set up parameters in new combined model
#         self.chromatic_models = new_models
#
#     def apply_operation_to_constituent_models(self, operation, *args, **kwargs):
#         """
#         Apply an operation to all models within a combined model
#         """
#         results = []
#         for m in self.chromatic_models.values():
#             try:
#                 # print(m, operation)
#                 op = getattr(m, operation)
#                 result = op(*args, **kwargs)
#                 if result is not None:
#                     results.append(result)
#             except Exception as e:
#                 print(f"Error applying {operation} to {m}: {e}")
#         if len(results) == 0:
#             return None
#         else:
#             return results
#
#     def summarize_parameters(self):
#         print(
#             "A CombinedModel itself does not have any parameters, however each of its constituent models do:\n"
#         )
#         self.apply_operation_to_constituent_models("summarize_parameters")
#
#     def attach_data(self, r):
#         """
#         Connect a `chromatic` Rainbow dataset to this object and the constituent models.
#         :parameter r:
#         Rainbow object
#         """
#         self.data = r._create_copy()
#         self.apply_operation_to_constituent_models("attach_data", r)
#
#     def choose_optimization_method(self, optimization_method="simultaneous"):
#         LightcurveModel.choose_optimization_method(self, optimization_method)
#         # self.apply_operation_to_constituent_models(
#         #     "choose_optimization_method", optimization_method
#         # )
#         for m in self.chromatic_models.values():
#             m.optimization = optimization_method
#
#         if optimization_method == "separate":
#             self.apply_operation_to_constituent_models("change_all_priors_to_Wavelike")
#
#     def setup_orbit(self):
#         """
#         Create an `exoplanet` orbit model, given the stored parameters.
#         """
#         self.apply_operation_to_constituent_models("setup_orbit")
#
#     def setup_lightcurves(self, store_models=False):
#         """
#         Set-up lightcurves in combined model : for each consituent model set-up the lightcurves according to their type
#         """
#
#         self.every_light_curve = {}
#         self.store_models = store_models
#         for cm in self.chromatic_models.values():
#             cm.store_models = store_models
#         # for each constituent model set-up the lightcurves according to their model type:
#         self.apply_operation_to_constituent_models("setup_lightcurves")
#
#         for i, mod in enumerate(self.chromatic_models.values()):
#             # for each lightcurve in the combined model, add/subtract/multiply/divide the lightcurve into the combined
#             # model
#             if i == 0:
#                 self.every_light_curve = add_dicts(
#                     self.every_light_curve, mod.every_light_curve
#                 )
#             else:
#                 self.every_light_curve = combination_options[
#                     self.how_to_combine[i - 1]
#                 ](self.every_light_curve, mod.every_light_curve)
#
#         if self.optimization == "separate":
#             models = self.pymc3_model
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#         else:
#             models = [self.pymc3_model]
#             datas = [self.get_data()]
#
#         if self.store_models:
#             # add a Deterministic parameter for easy extraction of the model later:
#             for i, (mod, data) in enumerate(zip(models, datas)):
#                 with mod:
#                     for w in range(data.nwave):
#                         k = f"wavelength_{i + w}"
#                         Deterministic(
#                             f"{self.name}_model_w{i + w}", self.every_light_curve[k]
#                         )
#
#     def get_results(self, **kw):
#         """
#         Extract the 'best-fit' parameter mean + error values from the summary table for each constituent model
#         """
#         results = []
#         for mod in self.chromatic_models.values():
#             results.append(mod.get_results(**kw))
#         # combine the results from all models:
#         df = pd.concat(results, axis=1)
#         # remove duplicated columns (wavelength):
#         df = df.loc[:, ~df.columns.duplicated()].copy()
#         return df
#
#     def get_model(self):
#         """
#         Extract each of the 'best-fit' models from the arviz summary table.
#         """
#         # can't use 'self.apply_operation_to_constituent_models("get_model")' because it returns the model
#         all_models, total_model = {}, {}
#         # for each constituent model get its 'best-fit' model:
#         for i, (name, m) in enumerate(self.chromatic_models.items()):
#             # get 'best-fit' model from constituent model:
#             model = m.get_model()
#             all_models[name] = model
#             # add this model to the total model:
#
#             if self.store_models == False:
#                 total_model = combination_options[self.how_to_combine[i - 1]](
#                     total_model, model
#                 )
#
#         if self.store_models:
#             # save the total model (for plotting/detrending):
#             all_models["total"] = self.extract_deterministic_model()  # total_model
#         else:
#             all_models["total"] = total_model
#
#         return all_models
#
#     def add_model_to_rainbow(self):
#         """
#         Add the 'best-fit' model to the `rainbow` object.
#         """
#         # if self.optimization == "separate":
#         #     datas = [self.get_data(i) for i in range(self.data.nwave)]
#         # else:
#         #     datas = [self.get_data()]
#
#         transit_model, systematics_model, total_model = {}, {}, {}
#         i_transit, i_sys = 0, 0
#         for i, mod in enumerate(self.chromatic_models.values()):
#             if isinstance(mod, TransitModel):
#                 if i_transit == 0:
#                     transit_model = mod.get_model()
#                 else:
#                     # I'm not sure that this works if the combination option is multiply or divide
#                     transit_model = combination_options[self.how_to_combine[i - 1]](
#                         transit_model, mod.get_model()
#                     )
#                 i_transit += 1
#             else:
#                 if i_sys == 0:
#                     systematics_model = mod.get_model()
#                 else:
#                     systematics_model = combination_options[self.how_to_combine[i - 1]](
#                         systematics_model, mod.get_model()
#                     )
#                 i_sys += 1
#
#             if i == 0:
#                 total_model = mod.get_model()
#             else:
#                 total_model = combination_options[self.how_to_combine[i - 1]](
#                     total_model, mod.get_model()
#                 )
#
#         # add the models to the rainbow object:
#         if self.outlier_flag:
#             data = self.data_without_outliers
#         else:
#             data = self.data
#
#         if self.optimization == "white_light":
#             data = self.white_light
#
#         r_with_model = data.attach_model(
#             model=np.array(list(total_model.values())),
#             planet_model=np.array(list(transit_model.values())),
#             systematics_model=np.array(list(systematics_model.values())),
#         )
#
#         # save the rainbow_with_model for plotting:
#         self.data_with_model = r_with_model
#
#     def make_transmission_spectrum_table(self, **kw):
#         results = self.apply_operation_to_constituent_models(
#             "make_transmission_spectrum_table", **kw
#         )
#         return results
#
#
# class PolynomialModel(LightcurveModel):
#     """
#     A polynomial model for the lightcurve.
#     """
#
#     def __init__(self, degree, independant_variable="time", name="polynomial", **kw):
#         """
#         Initialize the polynomial model.
#
#         :parameter degree (integer): the degree of the polynomial
#         :parameter independant_variable (str): the independant variable of the polynomial (default = time)
#         :parameter name (str): the name of the model (default = "polynomial")
#         :parameter kw: keyword arguments for initialising the chromatic model
#         """
#         # only require the constant term:
#         self.required_parameters = ["p_0"]
#
#         super().__init__(**kw)
#         self.degree = degree
#         self.independant_variable = independant_variable
#         self.set_defaults()
#         self.metadata = {}
#
#         # if name is not None:
#         # self.required_parameters = [f"{name}_{a}" for a in self.required_parameters]
#         # self.defaults = add_string_before_each_dictionary_key(self.defaults, name)
#         self.set_name(name)
#
#     def __repr__(self):
#         """
#         Print the polynomial model.
#         """
#         return "<chromatic polynomial model 🌈>"
#
#     def set_name(self, name):
#         """
#         Set the name of the model.
#         """
#         self.name = name
#
#     def set_defaults(self):
#         """
#         Set the default parameters for the model.
#         """
#         for d in range(self.degree + 1):
#             try:
#                 self.defaults = self.defaults | {f"p_{d}": 0.0}
#             except TypeError:
#                 # the | dictionary addition is only in Python 3.9
#                 self.defaults = {**self.defaults, **{f"p_{d}": 0.0}}
#
#     # def get_prior(self, i, *args, **kwargs):
#     #     data = self.get_data()
#     #     # self.degree = self.parameters["p"].inputs["shape"] - 1
#     #     x = data.time.to_value("day")
#     #     poly = []
#     #
#     #     p = self.parameters["p"].get_prior(i)
#     #
#     #     for d in range(self.degree + 1):
#     #         # p = self.parameters[f"p_{d}"].get_prior(i)
#     #         # fluxlike: x[i]
#     #         # x = self.data.get(fluxlike_thing:str)[i_wave,:]
#     #         poly.append(p[d] * (x**d))
#     #
#     #     return pm.math.sum(poly, axis=0)
#
#     def setup_lightcurves(self, store_models=False):
#         """
#         Create a polynomial model, given the stored parameters.
#         [This should be run after .attach_data()]
#         """
#
#         if self.optimization == "separate":
#             models = self.pymc3_model
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#         else:
#             models = [self.pymc3_model]
#             datas = [self.get_data()]
#
#         # if the model has a name then add this to each parameter's name
#         if hasattr(self, "name"):
#             name = self.name + "_"
#         else:
#             name = ""
#
#         if not hasattr(self, "every_light_curve"):
#             self.every_light_curve = {}
#
#         if store_models == True:
#             self.store_models = store_models
#
#         for j, (mod, data) in enumerate(zip(models, datas)):
#             with mod:
#                 for i, w in enumerate(data.wavelength):
#                     poly = []
#
#                     # get the independent variable from the Rainbow object:
#                     x = data.get(self.independant_variable)
#                     if len(np.shape(x)) > 1:
#                         x = x[i, :]
#                     # if the independant variable is time, convert to days:
#                     if self.independant_variable == "time":
#                         x = x.to_value("day")
#                     else:
#                         try:
#                             x = x.to_value()
#                         except AttributeError:
#                             pass
#
#                     # normalise the x values and store the mean/std values in metadata
#                     self.metadata["mean_" + self.independant_variable] = np.mean(x)
#                     self.metadata["std_" + self.independant_variable] = np.std(x)
#                     x = (x - np.mean(x)) / np.std(x)
#                     self.independant_variable_normalised = x
#
#                     # compute the polynomial:
#                     for d in range(self.degree + 1):
#                         p = self.parameters[f"{name}p_{d}"].get_prior(i + j)
#                         poly.append(p * (x**d))
#
#                     if self.store_models:
#                         # add a Deterministic parameter to the model for easy extraction later:
#                         Deterministic(
#                             f"{name}model_w{i + j}", pm.math.sum(poly, axis=0)
#                         )
#
#                     # add the polynomial to the overall lightcurve:
#                     if f"wavelength_{i + j}" not in self.every_light_curve.keys():
#                         self.every_light_curve[f"wavelength_{i + j}"] = pm.math.sum(
#                             poly, axis=0
#                         )
#                     else:
#                         self.every_light_curve[f"wavelength_{i + j}"] += pm.math.sum(
#                             poly, axis=0
#                         )
#
#     def polynomial_model(self, poly_params, i=0):
#         """
#         Create a polynomial model, given the passed parameters.
#         :parameter poly_params (dict): the parameters of the polynomial
#         """
#         poly = []
#
#         if self.optimization == "separate":
#             data = self.get_data(i)
#         else:
#             data = self.get_data()
#
#         if hasattr(self, "independant_variable_normalised"):
#             x = self.independant_variable_normalised
#         else:
#             # get the independent variable from the Rainbow object:
#             x = data.get(self.independant_variable)
#             # if the independant variable is time, convert to days:
#             if self.independant_variable == "time":
#                 x = x.to_value("day")
#
#         if len(np.shape(x)) > 1:
#             x = x[i, :]
#
#         try:
#             for d in range(self.degree + 1):
#                 poly.append(poly_params[f"{self.name}_p_{d}"] * (x**d))
#             return np.sum(poly, axis=0)
#         except KeyError:
#             for d in range(self.degree + 1):
#                 poly.append(poly_params[f"{self.name}_p_{d}_w0"] * (x**d))
#             return np.sum(poly, axis=0)
#
#     def get_model(self, as_dict=True, as_array=False):
#         """
#         Return the 'best-fit' model from the summary table as a dictionary or as an array
#         """
#
#         if self.optimization == "separate":
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#         else:
#             datas = [self.get_data()]
#
#         if self.store_models:
#             return LightcurveModel.get_model(as_dict=as_dict, as_array=as_array)
#         else:
#             model = {}
#             for i, data in enumerate(datas):
#                 poly_params = self.extract_from_posteriors(self.summary, i)
#                 model_i = self.polynomial_model(poly_params, i)
#                 model[f"w{i}"] = model_i
#             if as_array:
#                 return np.array(list(model.values()))
#             elif as_dict:
#                 return model
#
#     def add_model_to_rainbow(self):
#         """
#         Add the polynomial model to the Rainbow object.
#         """
#         if self.outlier_flag:
#             data = self.data_without_outliers
#         else:
#             data = self.data
#
#         if self.optimization == "white_light":
#             data = self.white_light
#
#         model = self.get_model(as_array=True)
#         r_with_model = data.attach_model(model=model, systematics_model=model)
#         self.data_with_model = r_with_model
#
#     # def imshow_with_models(self, **kw):
#     #     """
#     #     Imshow the Rainbow with the model.
#     #     """
#     #     model = self.get_model(as_array=True)
#     #     r_with_model = self.data.attach_model(model=model, systematics_model=model)
#     #     r_with_model.imshow_with_models(**kw)
#
#
# class TransitModel(LightcurveModel):
#     """
#     A transit model for the lightcurve.
#     """
#
#     def __repr__(self):
#         """
#         Print the transit model.
#         """
#         return "<chromatic transit model 🌈>"
#
#     def __init__(self, name="transit", **kw):
#         """
#         Initialise the transit model.
#         """
#
#         self.required_parameters = [
#             "stellar_radius",
#             "stellar_mass",
#             "radius_ratio",
#             "period",
#             "epoch",
#             "baseline",
#             "impact_parameter",
#             "limb_darkening",
#         ]
#
#         super().__init__(**kw)
#         self.set_defaults()
#         self.set_name(name)
#
#     def set_defaults(self):
#         """
#         Set the default parameters for the model.
#         """
#         self.defaults = dict(
#             stellar_radius=1.0,
#             stellar_mass=1.0,
#             radius_ratio=1.0,
#             period=1.0,
#             epoch=0.0,
#             baseline=1.0,
#             impact_parameter=0.5,
#             eccentricity=0.0,
#             omega=np.pi / 2.0,
#             limb_darkening=[0.2, 0.2],
#         )
#
#     def set_name(self, name):
#         """
#         Set the name of the model.
#         :paramter name (str): the name of the model
#         """
#         self.name = name
#
#     def setup_orbit(self):
#         """
#         Create an `exoplanet` orbit model, given the stored parameters.
#         [This should be run after .setup_parameters()]
#         """
#
#         # if the optimization method is separate wavelengths then set up for looping
#         if self.optimization == "separate":
#             models = self.pymc3_model
#         else:
#             models = [self.pymc3_model]
#
#         # if the model has a name then add this to each parameter"s name
#         if hasattr(self, "name"):
#             name = self.name + "_"
#         else:
#             name = ""
#             print("No name set for the model.")
#         # print(name)
#
#         self.orbit = []
#         for j, mod in enumerate(models):
#             with mod:
#
#                 # Set up a Keplerian orbit for the planets.py
#                 orbit = xo.orbits.KeplerianOrbit(
#                     period=self.parameters[name + "period"].get_prior(j),
#                     t0=self.parameters[name + "epoch"].get_prior(j),
#                     b=self.parameters[name + "impact_parameter"].get_prior(j),
#                     # ecc=self.parameters[name + "eccentricity"].get_prior(j),
#                     # omega=self.parameters[name + "omega"].get_prior(j),
#                     r_star=self.parameters[name + "stellar_radius"].get_prior(j),
#                     m_star=self.parameters[name + "stellar_mass"].get_prior(j),
#                 )
#
#                 # store a separate orbit for each model if we are optimizing wavelengths separately
#                 if self.optimization == "separate":
#                     self.orbit.append(orbit)
#                 else:
#                     self.orbit = orbit
#
#     # def get_prior(self, i):
#     #
#     #     # ensure that attach data has been run before setup_lightcurves
#     #     if not hasattr(self, "data"):
#     #         print("You need to attach some data to this chromatic model!")
#     #         return
#     #
#     #     # ensure that setup_orbit has been run before setup_lightcurves
#     #     if not hasattr(self, "orbit"):
#     #         self.setup_orbit()
#     #
#     #     limb_darkening = self.parameters["limb_darkening"].get_prior(i)
#     #     light_curves = xo.LimbDarkLightCurve(limb_darkening).get_light_curve(
#     #         orbit=self.orbit,
#     #         r=self.parameters["radius_ratio"].get_prior(i)
#     #         * self.parameters["stellar_radius"].get_prior(),
#     #         t=list(self.data.time.to_value("day")),
#     #     )
#     #     return pm.math.sum(
#     #         light_curves, axis=-1
#     #     )  # + (self.parameters["baseline"].get_prior(i))
#
#     def setup_lightcurves(self, store_models=False):
#         """
#         Create an `exoplanet` light curve model, given the stored parameters.
#         [This should be run after .attach_data()]
#         """
#
#         # ensure that attach data has been run before setup_lightcurves
#         if not hasattr(self, "data"):
#             print("You need to attach some data to this chromatic model!")
#             return
#
#         # ensure that setup_orbit has been run before setup_lightcurves
#         if not hasattr(self, "orbit"):
#             self.setup_orbit()
#
#         # if the model has a name then add this to each parameter"s name
#         if hasattr(self, "name"):
#             name = self.name + "_"
#         else:
#             name = ""
#             print("No name set for the model.")
#
#         # set up the models, data and orbits in a format for looping
#         if self.optimization == "separate":
#             models = self.pymc3_model
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#             orbits = self.orbit
#         else:
#             models = [self.pymc3_model]
#             datas = [self.get_data()]
#             orbits = [self.orbit]
#
#         if not hasattr(self, "every_light_curve"):
#             self.every_light_curve = {}
#
#         if store_models == True:
#             self.store_models = store_models
#
#         for j, (mod, data, orbit) in enumerate(zip(models, datas, orbits)):
#             with mod:
#                 for i, w in enumerate(data.wavelength):
#                     # create quadratic limb-darkening lightcurves from Exoplanet
#                     limb_darkening = self.parameters[name + "limb_darkening"].get_prior(
#                         j + i
#                     )
#                     planet_radius = self.parameters[name + "radius_ratio"].get_prior(
#                         j + i
#                     ) * self.parameters[name + "stellar_radius"].get_prior(j + i)
#                     light_curves = xo.LimbDarkLightCurve(
#                         limb_darkening
#                     ).get_light_curve(
#                         orbit=orbit,
#                         r=planet_radius,
#                         t=list(data.time.to_value("day")),
#                     )
#
#                     Deterministic(
#                         name + "a_R*",
#                         orbit.a
#                         / self.parameters[name + "stellar_radius"].get_prior(j + i),
#                     )
#
#                     # calculate the transit + baseline model
#                     mu = pm.math.sum(light_curves, axis=-1) + (
#                         self.parameters[name + "baseline"].get_prior(j + i)
#                     )
#
#                     if self.store_models:
#                         # add a Deterministic parameter to the model for easy extraction later
#                         Deterministic(f"{name}model_w{i + j}", mu)
#
#                     # add the transit to the light curve
#                     if f"wavelength_{j + i}" not in self.every_light_curve.keys():
#                         self.every_light_curve[f"wavelength_{j + i}"] = pm.math.sum(
#                             light_curves, axis=-1
#                         ) + (self.parameters[name + "baseline"].get_prior(j + i))
#                     else:
#                         self.every_light_curve[f"wavelength_{j + i}"] += pm.math.sum(
#                             light_curves, axis=-1
#                         ) + (self.parameters[name + "baseline"].get_prior(j + i))
#
#                 self.model_chromatic_transit_flux = [
#                     self.every_light_curve[k] for k in tqdm(self.every_light_curve)
#                 ]
#
#     def transit_model(self, transit_params, i=0):
#         """
#         Create a transit model given the passed parameters.
#         :parameter transit_params (dict): A dictionary of parameters to be used in the transit model.
#         """
#         if self.optimization == "separate":
#             data = self.get_data(i)
#         else:
#             data = self.get_data()
#         name = self.name
#         orbit = xo.orbits.KeplerianOrbit(
#             period=transit_params[f"{name}_period"],
#             t0=transit_params[f"{name}_epoch"],
#             b=transit_params[f"{name}_impact_parameter"],
#             # ecc=transit_params["eccentricity"],
#             # omega=transit_params["omega"],
#             r_star=transit_params[f"{name}_stellar_radius"],
#             m_star=transit_params[f"{name}_stellar_mass"],
#         )
#         # try:
#         ldlc = (
#             xo.LimbDarkLightCurve(transit_params[f"{name}_limb_darkening"])
#             .get_light_curve(
#                 orbit=orbit,
#                 r=transit_params[f"{name}_radius_ratio"]
#                 * transit_params[f"{name}_stellar_radius"],
#                 t=list(data.time.to_value("day")),
#             )
#             .eval()
#         )
#         return ldlc.transpose()[0] + transit_params[f"{name}_baseline"]
#         # except KeyError:
#         #     ldlc = (
#         #         xo.LimbDarkLightCurve(transit_params[f"{name}_limb_darkening_w0"])
#         #         .get_light_curve(
#         #             orbit=orbit,
#         #             r=transit_params[f"{name}_radius_ratio_w0"]
#         #             * transit_params[f"{name}_stellar_radius"],
#         #             t=list(data.time.to_value("day")),
#         #         )
#         #         .eval()
#         #     )
#         #     return ldlc.transpose()[0] + transit_params[f"{name}_baseline_w0"]
#
#     def plot_orbit(self, timedata=None):
#         """
#         Plot the orbit model.
#         :parameter timedata (array): An array of times to plot the orbit at (assuming that .attach_data() hasn't been
#         run yet).
#         """
#
#         # If the data hasn't been attached yet, then use the timedata passed to the function
#         if not hasattr(self, "data"):
#             if timedata is None:
#                 warnings.warn(
#                     "No data attached to this object and no time data provided. Plotting orbit will not work."
#                 )
#                 print(
#                     "No data attached to this object and no time data provided. Plotting orbit will not work."
#                 )
#                 return
#         else:
#             timedata = self.data.time
#
#         # if the optimization method is separate wavelengths then set up for looping
#         if self.optimization == "separate":
#             models = self.pymc3_model
#             orbits = self.orbit
#         else:
#             models = [self.pymc3_model]
#             orbits = [self.orbit]
#
#         # plot the orbit
#         for j, (mod, orbit) in enumerate(zip(models, orbits)):
#             with mod:
#                 x, y, z = [
#                     eval_in_model(bla, point=mod.test_point)
#                     for bla in orbit.get_planet_position(timedata)
#                 ]  # {}
#                 plt.figure(figsize=(10, 3))
#                 theta = np.linspace(0, 2 * np.pi)
#                 plt.fill_between(np.cos(theta), np.sin(theta), color="gray")
#                 plt.scatter(x, y, c=timedata)
#                 plt.axis("scaled")
#                 plt.ylim(-1, 1)
#                 plt.show()
#                 plt.close()
#
#     def get_model(self, as_dict=True, as_array=False):
#         """
#         Return the 'best-fit' model from the summary table as a dictionary or as an array
#         """
#         if self.optimization == "separate":
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#         else:
#             datas = [self.get_data()]
#
#         if self.store_models:
#             return LightcurveModel.get_model(as_dict=as_dict, as_array=as_array)
#         else:
#             model = {}
#             for i, data in enumerate(datas):
#                 transit_params = self.extract_from_posteriors(self.summary, i)
#                 model_i = self.transit_model(transit_params, i)
#                 model[f"w{i}"] = model_i
#             if as_array:
#                 return np.array(list(model.values()))
#             elif as_dict:
#                 return model
#
#     def add_model_to_rainbow(self):
#         """
#         Add the transit model to the Rainbow object.
#         """
#         if self.outlier_flag:
#             data = self.data_without_outliers
#         else:
#             data = self.data
#
#         if self.optimization == "white_light":
#             data = self.white_light
#
#         model = self.get_model(as_array=True)
#         r_with_model = data.attach_model(model=model, planet_model=model)
#         self.data_with_model = r_with_model
#
#     # def plot_lightcurves(self, summary=None, trace=None, ax=None, **kw):
#     #     if ax is None:
#     #         ax = plt.subplot()
#     #     plt.sca(ax)
#     #
#     #     data = self.get_data()
#     #
#     #     if summary is not None:
#     #         params = [
#     #             "period",
#     #             "epoch",
#     #             "impact_parameter",
#     #             "stellar_radius",
#     #             "stellar_mass",
#     #             "stellar_mass",
#     #             "limb_darkening",
#     #             "radius_ratio",
#     #         ]
#     #         param_dict = {}
#     #         posterior_means = summary["mean"]
#     #
#     #         for p in params:
#     #             if self.parameters[p] in posterior_means.index:
#     #                 param_dict[p] = posterior_means[p]
#     #             else:
#     #                 param_dict[p] = self.parameters[p].value
#     #
#     #         orbit = xo.orbits.KeplerianOrbit(
#     #             period=param_dict["period"],
#     #             t0=param_dict["epoch"],
#     #             b=param_dict["impact_parameter"],
#     #             r_star=param_dict["stellar_radius"],
#     #             m_star=param_dict["stellar_mass"],
#     #         )
#     #         xo.LimbDarkLightCurve(param_dict["limb_darkening"]).get_light_curve(
#     #             orbit=self.orbit,
#     #             r=param_dict["radius_ratio"] * param_dict["stellar_radius"],
#     #             t=list(data.time.to_value("day")),
#     #         )
#     #
#     #         if trace is not None:
#     #             ndraws = 50
#     #             posterior_traces = self.sample_posterior(ndraws=ndraws)
#     #             plt.plot(posterior_traces)
#     #             for i in range(ndraws):
#     #                 flux_for_this_sample = np.array(
#     #                     [
#     #                         posterior_traces[f"wavelength_{w}_data"][i]
#     #                         for w in range(data.nwave)
#     #                     ]
#     #                 )
#     #                 plt.plot(data.time, flux_for_this_sample)
#     #
#     #     data.plot(ax=ax, **kw)
