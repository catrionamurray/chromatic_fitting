from ..imports import *


def check_rainbow(r):
    try:
        assert r.nwave > 0
        assert r.ntime > 0
    except Exception as e:
        print(
            f"Checking Rainbow {r} failed with error: {e}"
            f"\nPotentially this is an issue with setting the ok wavelengths for your Rainbow? "
            f"Try removing that step and running again!"
        )
    return


def check_model(mod):
    # include some model check
    return


def check_results(result):
    # include some result check
    return


def check_initial_guess(model):
    try:
        assert np.all(np.isfinite(model.check_test_point().values))
    except Exception as e:
        print(
            f"Checking test point for model failed with error: {e}"
            f"\n Likely because one of your priors is unbounded..."
        )
        for i in model.check_test_point().iteritems():
            if not np.isfinite(i[1]):
                print(
                    f"{i[0]} does not have a finite logP value ({i[1]}) for initial guess!"
                )
    return
