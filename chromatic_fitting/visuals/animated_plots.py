from ..imports import *


def get_animation_writer_and_displayer(filename="animation.html", **kw):
    """
    Create the right animation writer based on filename.

    Parameters
    ----------
    filename : str
        The filename of the movie to create.

    Returns
    -------
    writer : MovieWriter
        The matplotlib writer object.
    displayer : ?
        The
    """

    # define the options
    writers = {"html": ani.HTMLWriter, "mp4": ani.FFMpegWriter, "gif": ani.PillowWriter}
    warnings = {
        "html": "Please try `pip insall matplotlib --upgrade` and rerunning?",
        "mp4": "Please try `conda install ffmpeg` and rerunning?",
        "gif": "Please try `pip insall matplotlib --upgrade` and rerunning?",
    }
    from IPython.display import HTML, Video, Image

    displayers = {"html": HTML, "mp4": Video, "gif": Image}

    # get the writer object
    suffix = filename.split(".")[-1]
    writer = writers[suffix](**kw)
    displayer = displayers[suffix]

    if writer.isAvailable():
        return writer, displayer
    else:
        raise ValueError(
            f"""
        The writer {writer} needed for your `.{suffix}` file is not available.
        {warnings[suffix]}
        """
        )


def setup_animated_scatter(
    self, ax=None, figurekw={}, scatterkw={}, textkw={}, modelkw={}, speckw={}
):
    """
    Wrapper to set up the basics of animate-able plot.

    This works for any general plot that has a single-color
    line or set of points (using `plt.plot`), with a text
    label in the upper right corner.

    Parameters
    ----------
    ax : Axes, optional
        The axes into which the plot should be drawn.
        If None, a new one will be created.
    figurekw : dict, optional
        A dictionary of keywords to be passed to `plt.figure`
    scatterkw : dict, optional
        A dictionary of keywords to be passed to `plt.scatter`
    textkw : dict, optional
        A dictionary of keywords to be passed to `plt.text`
    """

    # make sure the ax and figure are defined
    if ax is None:
        kw = dict(facecolor="white")
        kw.update(**figurekw)
        fig, ax = plt.subplots(**kw)
    else:
        fig = ax.get_figure()

    plt.sca(ax)

    this_scatterkw = dict(c=[], cmap=self.cmap, norm=self.norm)
    this_scatterkw.update(**scatterkw)
    scatter = plt.scatter([], [], **this_scatterkw)

    this_textkw = dict(
        x=0.98, y=0.96, s="", ha="right", va="top", transform=ax.transAxes
    )
    this_textkw.update(**textkw)
    text = plt.text(**this_textkw)

    this_modelkw = dict()
    this_modelkw.update(**modelkw)
    model = plt.plot([], [], linestyle="-", c="k", alpha=0.3, **this_modelkw)

    this_speckw = dict()  # c=[], cmap=self.cmap)
    this_speckw.update(**speckw)
    spec = plt.scatter([], [], c="k", **this_speckw)
    #     _,_,(bottoms,tops) = plt.errorbar([],[],[])
    #     spec_err = plt.plot([],[],linestyle='None', c='k',alpha=0)

    plt.title(self.get("title"))

    # return a dictionary with things that will be useful to hang onto
    return dict(fi=fig, ax=ax, scatter=scatter, text=text, model=model, spec=spec)


def setup_animate_transmission_spectrum(
    self,
    transmission_spectrum,
    ax=None,
    quantity="flux",
    orientation="vertical",
    xlim=[None, None],
    ylim=[None, None],
    cmap=None,
    vmin=None,
    vmax=None,
    ylabel=None,
    scatterkw={},
    textkw={},
    modelkw={},
    speckw={},
):
    """
    Setup an animation to how the lightcurve changes
    as we flip through every wavelength.

    Parameters
    ----------
    ax : Axes, optional
        The axes into which this animated plot should go.
    quantity : string, optional
        Which fluxlike quantity should be retrieved? (default = 'flux')
    xlim : tuple, optional
        Custom xlimits for the plot
    ylim : tuple, optional
        Custom ylimits for the plot
    cmap : str, Colormap, optional
        The color map to use for expressing wavelength
    vmin : Quantity, optional
        The minimum value to use for the wavelength colormap
    vmax : Quantity, optional
        The maximum value to use for the wavelength colormap
    scatterkw : dict, optional
        A dictionary of keywords to be passed to `plt.scatter`
        so you can have more detailed control over the plot
        appearance. Common keyword arguments might include:
        `[s, c, marker, alpha, linewidths, edgecolors, zorder]` (and more)
        More details are available at
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    textkw : dict
        A dictionary of keywords passed to `plt.text`
        so you can have more detailed control over the text
        appearance. Common keyword arguments might include:
        `[alpha, backgroundcolor, color, fontfamily, fontsize,
          fontstyle, fontweight, rotation, zorder]` (and more)
        More details are available at
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
    """

    # self._make_sure_cmap_is_defined(cmap=cmap, vmin=vmin, vmax=vmax)

    with quantity_support():

        rr = transmission_spectrum["transit_radius_ratio"].values
        rr_err = [
            [r_n, r_p]
            for r_n, r_p in zip(
                transmission_spectrum["transit_radius_ratio_neg_error"].values,
                transmission_spectrum["transit_radius_ratio_pos_error"].values,
            )
        ]
        #         print(rr_err)
        wavelength = self.wavelength  # transmission_spectrum['wavelength']
        ax[2].plot(wavelength, rr, "k", alpha=0.1)

        ylims = [
            [
                0.995 * np.nanmin(self.get(quantity)),
                1.005 * np.nanmax(self.get(quantity)),
            ],
            [
                0.995 * np.nanmin(self.get(quantity)),
                1.005 * np.nanmax(self.get(quantity)),
            ],
            [0.99 * np.nanmin(rr), 1.01 * np.nanmax(rr)],
        ]

        xlims = [
            [np.nanmin(self.time), np.nanmax(self.time)],
            [np.nanmin(self.time), np.nanmax(self.time)],
            [0.99 * np.nanmin(wavelength), 1.01 * np.nanmax(wavelength)],
        ]

        xlabels = [
            f"{self._time_label} ({self.time.unit.to_string('latex_inline')})",
            f"{self._time_label} ({self.time.unit.to_string('latex_inline')})",
            f"{self._wave_label} ({self.wavelength.unit.to_string('latex_inline')})",
        ]

        ylabels = [quantity, quantity, "Radius Ratio"]

        titles = [
            "Lightcurve",
            "Systematics-Detrended Lightcurve",
            "Transmission Spectrum",
        ]

        self._animate_lightcurves_components = []

        for i, ax_i in enumerate(ax):
            # keep track of the things needed for the animation
            self._animate_lightcurves_components.append(
                setup_animated_scatter(
                    self,
                    ax=ax_i,
                    scatterkw=scatterkw,
                    textkw=textkw,
                    modelkw=modelkw,
                    speckw=speckw,
                )
            )
            #             print(self._animate_lightcurves_components)

            ax_i = self._animate_lightcurves_components[i]["ax"]

            # set the plot limits
            ax_i.set_xlim(xlims[i][0], xlims[i][1])
            ax_i.set_ylim(ylims[i][0], ylims[i][1])

            # set the axis labels
            ax_i.set_xlabel(xlabels[i])
            ax_i.set_ylabel(ylabels[i])

            ax_i.set_title(titles[i])

            # guess a good number of digits to round
            ndigits = np.minimum(
                int(np.floor(np.log10(np.min(np.diff(self.wavelength)).value))), 0
            )
            format_code = f"{{:.{-ndigits}f}}"

        def update(frame):
            """
            This function will be called to update each frame
            of the animation.

            Parameters
            ----------
            frame : int
                An integer that will advance with each frame.
            """

            # pull out the x and y values to plot
            x, y, _ = self.get_ok_data_for_wavelength(frame, y=quantity)

            x_model = self.time
            y_model = self.fluxlike["model"][frame]
            y_model_planet = self.fluxlike["planet_model"][frame]
            try:
                y_model_sys = self.fluxlike["systematics_model"][frame]
            except:
                y_model_sys = np.ones(len(y))
            # y_model_sys = self.fluxlike["systematics_model"][frame]

            # x = self.time
            # y = self.flux[frame]
            c = self.wavelength[frame].to("micron").value * np.ones(self.ntime)

            # update the label in the corner
            self._animate_lightcurves_components[0]["text"].set_text(
                f"w = {format_code.format(self.wavelength[frame].value)} {self.wavelength.unit.to_string('latex')}"
            )
            self._animate_lightcurves_components[1]["text"].set_text(
                f"w = {format_code.format(self.wavelength[frame].value)} {self.wavelength.unit.to_string('latex')}"
            )

            # update the plot data
            self._animate_lightcurves_components[0]["scatter"].set_offsets(
                np.transpose([x, y])
            )
            self._animate_lightcurves_components[0]["scatter"].set_array(c)

            # update the plot data
            self._animate_lightcurves_components[1]["scatter"].set_offsets(
                np.transpose([x, y / y_model_sys])
            )
            self._animate_lightcurves_components[1]["scatter"].set_array(c)

            # update the model data
            self._animate_lightcurves_components[0]["model"][0].set_data(x, y_model)
            self._animate_lightcurves_components[1]["model"][0].set_data(
                x, y_model_planet
            )

            # update transmission spectrum
            self._animate_lightcurves_components[2]["spec"].set_offsets(
                np.transpose(
                    [self.wavelength[: frame + 1].to("micron").value, rr[: frame + 1]]
                )
            )
            #             c2 = self.wavelength[frame].to("micron").value #* np.ones(self.ntime)
            #             self._animate_lightcurves_components[2]["spec"].set_array(self.wavelength[:frame+1].to("micron").value)#c)
            #             self._animate_lightcurves_components[2]["spec"]
            #             speckw['c'] = self.wavelength[frame].to("micron").value

            #             print(c[frame], rr[frame])
            #             print(self._animate_lightcurves_components[2]["spec_err"])
            #             print(frame, np.shape(rr_err))
            #             self._animate_lightcurves_components[2]["spec_err"][0].set_data(
            #                 [self.wavelength[frame].to("micron").value,self.wavelength[frame].to("micron").value],
            #                 [rr[frame]-rr_err[frame][0],rr[frame]+rr_err[frame][1]],
            #             )

            return [
                (alc["text"], alc["scatter"], alc["model"], alc["spec"])
                for alc in self._animate_lightcurves_components
            ]

        # hold onto this update function in case we need it elsewhere
        upd = update
        for i in range(len(ax)):
            self._animate_lightcurves_components[i]["update"] = upd


def animate_transmission_spectrum(
    self,
    transmission_spectrum,
    filename="animated-transmission-spectrum.gif",
    fps=4,
    dpi=None,
    bitrate=None,
    **kwargs,
):
    """
    Create an animation to show how the lightcurve changes
    as we flip through every wavelength.

    Parameters
    ----------
    filename : str
        Name of file you'd like to save results in.
        Currently supports only .gif or .html files.
    fps : float
        frames/second of animation
    ax : Axes
        The axes into which this animated plot should go.
    xlim : tuple
        Custom xlimits for the plot
    ylim : tuple
        Custom ylimits for the plot
    cmap : str,
        The color map to use for expressing wavelength
    vmin : Quantity
        The minimum value to use for the wavelength colormap
    vmax : Quantity
        The maximum value to use for the wavelength colormap
    scatterkw : dict
        A dictionary of keywords to be passed to `plt.scatter`
        so you can have more detailed control over the plot
        appearance. Common keyword arguments might include:
        `[s, c, marker, alpha, linewidths, edgecolors, zorder]` (and more)
        More details are available at
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    textkw : dict
        A dictionary of keywords passed to `plt.text`
        so you can have more detailed control over the text
        appearance. Common keyword arguments might include:
        `[alpha, backgroundcolor, color, fontfamily, fontsize,
          fontstyle, fontweight, rotation, zorder]` (and more)
        More details are available at
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
    """
    my_dpi = 96
    fig, axes = plt.subplots(
        facecolor="w",
        figsize=(1920 / (1.7 * my_dpi), 1080 / (2.7 * my_dpi)),
        dpi=my_dpi,
    )
    #     plt.subplots_adjust(wspace=0.35)
    plt.subplots_adjust(hspace=0.5)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax4 = plt.subplot(2, 1, 2)
    ax = [ax1, ax2, ax4]
    rr = transmission_spectrum["transit_radius_ratio"].values
    rr_err = [
        transmission_spectrum["transit_radius_ratio_neg_error"].values,
        transmission_spectrum["transit_radius_ratio_pos_error"].values,
    ]
    ax4.plot(self.wavelength.to_value("micron"), rr, "k.", linestyle="None", alpha=0.05)
    ax4.errorbar(
        self.wavelength.to_value("micron"),
        rr,
        yerr=rr_err,
        c="k",
        linestyle="None",
        alpha=0.2,
        capsize=2,
        zorder=10,
    )

    setup_animate_transmission_spectrum(
        self,
        transmission_spectrum,
        ax=ax,
        scatterkw={"s": 3},
        speckw={"s": 5, "zorder": 1},
        **kwargs,
    )
    ax4.set_ylim(0.142, 0.149)
    ax4.set_xlim(0.8, 2.9)

    #     setup_animate_transmission_spectrum(self,
    #         0, animation=False, orientation=orientation, transmission_spectrum, ax=ax, **kwargs,
    #     )

    filename = self._label_plot_file(filename)

    # initialize the animator
    writer, displayer = get_animation_writer_and_displayer(
        filename=filename, fps=fps, bitrate=bitrate
    )

    # set up to save frames directly into the animation
    #     print(self._animate_lightcurves_components)
    for i, ax_i in enumerate(ax):
        figure = self._animate_lightcurves_components[i]["fi"]
        with writer.saving(figure, filename, dpi or figure.get_dpi()):
            for j in tqdm(range(self.nwave), leave=False):
                self._animate_lightcurves_components[i]["update"](j)
                #                 setup_animate_transmission_spectrum(self,
                #                     i, animation=True, orientation=orientation, transmission_spectrum, ax=ax, **kwargs,
                #                 )
                writer.grab_frame()

    # close the figure that was created
    plt.close(figure)

    # display the animation
    from IPython.display import display

    try:
        display(displayer(filename, embed=True))
    except TypeError:
        display(displayer(filename))
