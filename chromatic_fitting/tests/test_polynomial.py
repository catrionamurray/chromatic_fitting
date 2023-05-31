import unittest
from chromatic_fitting.models import *


class TestPolynomialModel(unittest.TestCase):
    def test_create_new_polynomial_model(self):
        # Create a new polynomial model
        model = PolynomialModel(degree=2)

        # Check if the model is an instance of PolynomialModel class
        self.assertIsInstance(model, PolynomialModel)

        # Check if the degree of the model is 2
        self.assertEqual(model.degree, 2)

        # Check if the model has the required_parameters attribute
        self.assertTrue(hasattr(model, "required_parameters"))
        self.assertTrue("p_0" in model.required_parameters)

        # Check if the model has the required_parameters attribute
        self.assertTrue(hasattr(model, "independant_variable"))
        self.assertTrue(model.independant_variable == "time")

        # Check if the model has the type_of_model attribute
        self.assertTrue(hasattr(model, "type_of_model"))
        self.assertTrue(model.type_of_model == "systematic")

        # Check if the model has the metadata attribute
        self.assertTrue(hasattr(model, "metadata"))

        # Check if the model has the xlims attribute
        self.assertTrue(hasattr(model, "xlims"))
        self.assertTrue(model.xlims is None)

        # Check if the model has the model attribute
        self.assertTrue(hasattr(model, "model"))
        self.assertTrue(model.model == model.polynomial_model)

        # Check if the model has the name attribute
        self.assertTrue(hasattr(model, "name"))
        self.assertTrue(model.name == "polynomial")

        # Check if the model has the defaults attribute
        self.assertTrue(hasattr(model, "defaults"))
        self.assertDictEqual(model.defaults, {"p_0": 0.0, "p_1": 0.0, "p_2": 0.0})

        # Check if the model has the every_light_curve attribute
        self.assertFalse(hasattr(model, "every_light_curve"))

        # Check if the model has the initial_guess attribute
        self.assertFalse(hasattr(model, "initial_guess"))

        # Check if the model has the store_models attribute
        self.assertTrue(hasattr(model, "store_models"))
        self.assertFalse(model.store_models)

    def test_xlims(self):
        model = PolynomialModel(degree=2, xlims=[2, 5])
        self.assertEqual(model.xlims, [2, 5])


if __name__ == "__main__":
    unittest.main()
