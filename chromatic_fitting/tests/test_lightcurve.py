import unittest

# from ..models.lightcurve import *
# from ..models.combined import *
from chromatic_fitting.models import *
from pymc3 import Uniform, Normal

from io import StringIO
import sys


class captured_output:
    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()

    def __enter__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self.stdout, self.stderr

    def __exit__(self, *args):
        self.stdout.close()
        self.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


class TestLightcurveModel(unittest.TestCase):
    def setUp(self):
        # initialize a LightcurveModel object with default parameters
        self.lc_model = LightcurveModel()

    def test_initialize_empty_model(self):
        # initialize an empty pymc3 model for the LightcurveModel
        self.lc_model.initialize_empty_model()
        self.assertIsInstance(self.lc_model._pymc3_model, pm.Model)

    def test_set_name(self):
        # set a new name for the model
        self.lc_model.set_name("new_name")
        # test if the name attribute has been changed
        self.assertEqual(self.lc_model.name, "new_name")

    def test_setup_parameters(self):
        # set up some parameters for the model
        name = self.lc_model.name
        if name != "":
            name = name + "_"

        self.lc_model.setup_parameters(
            param1=Fixed(2), param2=Fitted(Uniform, lower=0, upper=1)
        )
        # test if the parameters attribute has been set up correctly
        self.assertEqual(len(self.lc_model.parameters), 2)
        self.assertIsInstance(self.lc_model.parameters[f"{name}param1"], Fixed)
        self.assertIsInstance(self.lc_model.parameters[f"{name}param2"], Fitted)

    def test_summarize_parameters(self):
        name = self.lc_model.name
        if name != "":
            name = name + "_"
        # set up some parameters for the model
        self.lc_model.setup_parameters(
            param1=Fixed(2), param2=Fitted(Uniform, lower=0, upper=1)
        )
        # test if the summarize_parameters method prints the parameters
        with captured_output() as (out, _):
            self.lc_model.summarize_parameters()
            output = out.getvalue().strip()
            self.assertIn("param1", output)
            self.assertIn("param2", output)

    def test_add(self):
        # create a second LightcurveModel object
        lc_model2 = LightcurveModel(name="lc_model2")
        lc_model2.setup_parameters(
            param3=Fixed(10), param4=Fitted(Uniform, lower=-1, upper=1)
        )
        # test if the __add__ method returns a CombinedModel object
        self.assertIsInstance(self.lc_model + lc_model2, CombinedModel)

    def test_subtract(self):
        # create a second LightcurveModel object
        lc_model2 = LightcurveModel()
        lc_model2.setup_parameters(
            param3=Fixed(10), param4=Fitted(Uniform, lower=-1, upper=1)
        )
        # test if the __sub__ method returns a CombinedModel object
        self.assertIsInstance(self.lc_model - lc_model2, CombinedModel)

    def test_multiply(self):
        # create a second LightcurveModel object
        lc_model2 = LightcurveModel()
        lc_model2.setup_parameters(
            param3=Fixed(10), param4=Fitted(Uniform, lower=-1, upper=1)
        )
        # test if the __mul__ method returns a CombinedModel object
        self.assertIsInstance(self.lc_model * lc_model2, CombinedModel)

    def test_divide(self):
        # create a second LightcurveModel object
        lc_model2 = LightcurveModel()
        lc_model2.setup_parameters(
            param3=Fixed(10), param4=Fitted(Uniform, lower=-1, upper=1)
        )
        # test if the __truediv__ method returns a CombinedModel object
        self.assertIsInstance(self.lc_model / lc_model2, CombinedModel)


if __name__ == "__main__":
    unittest.main()
