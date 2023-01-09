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
