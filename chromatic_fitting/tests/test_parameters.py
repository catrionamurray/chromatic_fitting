import unittest
import pytest
from chromatic_fitting.parameters import *
from pymc3 import Normal
import pymc3 as pm


class TestParameter(unittest.TestCase):
    def test_set_name(self):
        param = Parameter()
        param.set_name("test_param")
        self.assertEqual(param.name, "test_param")


class TestFixed(unittest.TestCase):
    def test_get_prior(self):
        a = Fixed(1.0)
        b = Fixed(1)
        assert a.value == 1.0
        assert a.get_prior() == 1.0
        assert b.value == 1
        assert b.get_prior() == 1

    def test_get_prior_vector(self):
        a = Fixed(1.0)
        b = Fixed(1)
        assert a.get_prior_vector() == 1.0
        assert b.get_prior_vector() == 1

        values = np.array([1, 2, 3])
        param = WavelikeFixed(values=values)
        prior = param.get_prior_vector()
        self.assertTrue(np.array_equal(prior, values))

    def test_wavelikefixed(self):
        values = [1.0, 2.0, 3.0]
        a = WavelikeFixed(values)

        assert a.values == values
        assert a.get_prior(i=0) == values[0]
        assert a.get_prior_vector(i=1) == values[1]
        assert a.get_prior_vector() == values
        assert a.get_prior_vector(shape=len(values)) == values


class TestFitted(unittest.TestCase):
    def test_generate_pymc3(self):
        a = Fitted(Normal, mu=1.0, sigma=0.1, name="a")
        with pm.Model():
            prior = a.generate_pymc3()
        #         self.assertIsInstance(prior, Normal)
        self.assertIsInstance(prior.distribution, Normal)
        assert a.inputs["mu"] == 1.0
        assert a.inputs["sigma"] == 0.1
        assert a.inputs["name"] == "a"

    def test_get_prior(self):
        # *** TEST NORMAL CASE ***
        a = Fitted(Normal, mu=1.0, sigma=0.1, name="a")

        with pm.Model() as mod:
            prior = a.get_prior()
        self.assertIsInstance(prior.distribution, Normal)

        assert mod["a"] == a.get_prior()
        assert mod["a"] == a._pymc3_prior

        # *** TEST CRASH CASES ***
        # should crash as I've passed a string argument:
        b = Fitted(Normal, mu="hello", sigma=0.1, name="a")
        with pm.Model():
            with pytest.raises(ValueError):
                b.get_prior()

        # should crash as there's no name provided:
        c = Fitted(Normal, mu=1.0, sigma=0.1)
        with pm.Model():
            with pytest.raises(TypeError):
                c.get_prior()

    def test_clear_prior(self):
        param = Fitted(Normal, mu=0, sigma=1, name="a")
        param.clear_prior()
        self.assertFalse(hasattr(param, "_pymc3_prior"))


class TestWavelikeFitted(unittest.TestCase):
    def test_label(self):
        param = WavelikeFitted(Normal, mu=0, sigma=1, name="test_param")
        label = param.label(2)
        self.assertEqual(label, "test_param_w2")

    def test_generate_pymc3(self):
        param = WavelikeFitted(Normal, mu=0, sigma=1, shape=3, name="test_param")
        with pm.Model():
            prior = param.generate_pymc3(i=2)
        self.assertIsInstance(prior.distribution, Normal)

    def test_get_prior_vector(self):
        # test vectorized parameter
        a = WavelikeFitted(Normal, mu=1.0, sigma=0.1, name="a")

        assert a.distribution == Normal
        assert a.inputs["mu"] == 1.0
        assert a.inputs["sigma"] == 0.1
        assert a.inputs["name"] == "a"

        with pm.Model() as mod:
            a.get_prior_vector(shape=3)

        assert a._pymc3_prior
        assert a.inputs["shape"] == 3
        assert mod["a"] == a.get_prior_vector()
        assert mod["a"][0]
        assert mod["a"][1]
        assert mod["a"][2]
        with pytest.raises(IndexError):
            assert mod["a"][3]

        # test non-vectorized (separate) parameter
        a = WavelikeFitted(Normal, mu=1.0, sigma=0.1, name="a")
        models = [pm.Model() for i in range(3)]
        for i in range(3):
            with models[i]:
                a.get_prior_vector(i=i, shape=1)

        assert a._pymc3_priors
        assert a.inputs["shape"] == 1
        assert "a_w0" in a._pymc3_priors.keys()
        assert "a_w1" in a._pymc3_priors.keys()
        assert "a_w2" in a._pymc3_priors.keys()
        assert a._pymc3_priors["a_w0"] == a.get_prior_vector(i=0)
        assert a._pymc3_priors["a_w1"] == a.get_prior_vector(i=1)
        assert a._pymc3_priors["a_w2"] == a.get_prior_vector(i=2)
        assert models[0]["a"][0]
        assert models[1]["a"][0]
        assert models[2]["a"][0]
        with pytest.raises(TypeError):
            assert a.get_prior_vector(i=3)
        with pytest.raises(IndexError):
            assert models[0]["a"][1]

        # should crash as I've passed a string argument:
        b = Fitted(Normal, mu="hello", sigma=0.1, name="a")
        with pm.Model():
            with pytest.raises(ValueError):
                b.get_prior_vector()

        # should crash as there's no name provided:
        c = Fitted(Normal, mu=1.0, sigma=0.1)
        with pm.Model():
            with pytest.raises(TypeError):
                c.get_prior_vector()


if __name__ == "__main__":
    unittest.main()
