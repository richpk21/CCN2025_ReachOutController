import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import warnings
from matplotlib.collections import LineCollection



def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)



# was (b, t, h)
# (t, h)
def plot_hidden_trajectory(ax, h: np.ndarray, pca: PCA, args: dict ):
    title: str = "Hidden Layer Trajectory"
    if args["title"]:
        title = args["title"]

    comp= pca.components_

    if not ax:
        fig1, ax = plt.subplots()
    line = None
    start_points = []
    end_points = []
    for i in range(h.shape[0]):
        Xx = []
        Xy = []
        for j in range(h.shape[1]):
            # Xt = tsne_hidden.fit_transform(all_h[i][j])
            # Xt = comp @ all_h[i][j]
            Xt = comp @ h[i][j]
            Xx.append(Xt[0])
            Xy.append(Xt[1])
            # plt.plot(Xt[:,0], Xt[:,1])
            # plt.plot(Xt[0,0], Xt[0,1], "bo")
            # plt.plot(Xt[-1,0], Xt[-1,1], "go")
        Xx = np.array(Xx)
        Xy = np.array(Xy)
        start_points.append( [Xx[0], Xy[0]] )
        end_points.append( [Xx[-1], Xy[-1]])
        # plt.plot(Xx, Xy)
        # plt.plot(Xx[0], Xy[0], "bo")
        # plt.plot(Xx[-1], Xy[-1], "go")
        # -------------- Create and show plot --------------
        # Some arbitrary function that gives x, y, and color values
        t = np.arange(0, h.shape[1], 1)
        color = np.linspace(0, h.shape[1], h.shape[1])

        # Create a figure and plot the line on it
        # fig1, ax1 = plt.subplots()
        lines = colored_line(Xx, Xy, color, ax, linewidth=1, cmap="brg")
        # fig1.colorbar(lines)  # add a color legend
        # Set the axis limits and tick positions
        ax.set_title(title)

    if line:
        fig1.colorbar(lines)  # add a color legend

    if start_points and end_points:
        connected_end_points = end_points
        connected_end_points.append(end_points[0])
        start_points = np.array(start_points)
        end_points = np.array(end_points)
        connected_end_points = np.array(connected_end_points)

        mu_start = np.mean(start_points)
        mu_end = np.mean(end_points)

        sig_start = np.std(start_points)
        sig_end = np.std(end_points)


        ax.plot(start_points[:,0], start_points[:,1], "bo")
        # ax.plot(connected_end_points[:,0], connected_end_points[:,1], "g-")
        ax.plot(end_points[:,0], end_points[:,1], "go")
        # print(f"sig_end: {sig_end}")

        # estimator = EllipticEnvelope(support_fraction=1.0, contamination=0.25)
        # estimator.fit(end_points)

        # DecisionBoundaryDisplay.from_estimator(
        #     estimator,
        #     end_points,
        #     response_method="decision_function",
        #     plot_method="contour",
        #     levels=[0],
        #     colors="green",
        #     ax=ax,
        # )




# was (b, t, h)
# (t, h)
def plot_hidden_trajectory_with_go_cue(ax, h: np.ndarray, go_cue_time:int, pca: PCA, args: dict ):
    title: str = "Hidden Layer Trajectory"
    if args["title"]:
        title = args["title"]

    comp= pca.components_

    if not ax:
        fig1, ax = plt.subplots()
    line = None
    start_points = []
    end_points = []
    for i in range(h.shape[0]):
        Xx = []
        Xy = []
        for j in range(100):
            # Xt = tsne_hidden.fit_transform(all_h[i][j])
            # Xt = comp @ all_h[i][j]
            Xt = comp @ h[i][j]
            Xx.append(Xt[0])
            Xy.append(Xt[1])
            # plt.plot(Xt[:,0], Xt[:,1])
            # plt.plot(Xt[0,0], Xt[0,1], "bo")
            # plt.plot(Xt[-1,0], Xt[-1,1], "go")
        Xx = np.array(Xx)
        Xy = np.array(Xy)
        start_points.append( [Xx[0], Xy[0]] )
        end_points.append( [Xx[-1], Xy[-1]])
        # plt.plot(Xx, Xy)
        # plt.plot(Xx[0], Xy[0], "bo")
        # plt.plot(Xx[-1], Xy[-1], "go")
        # -------------- Create and show plot --------------
        # Some arbitrary function that gives x, y, and color values
        t = np.arange(0, 100, 1)
        color = np.linspace(0, 100, 100)

        # Create a figure and plot the line on it
        # fig1, ax1 = plt.subplots()
        lines = colored_line(Xx, Xy, color, ax, linewidth=1, cmap="brg")
        # fig1.colorbar(lines)  # add a color legend
        # Set the axis limits and tick positions
        ax.set_title(title)

    if line:
        fig1.colorbar(lines)  # add a color legend

    if start_points and end_points:
        connected_end_points = end_points
        connected_end_points.append(end_points[0])
        start_points = np.array(start_points)
        end_points = np.array(end_points)
        connected_end_points = np.array(connected_end_points)

        mu_start = np.mean(start_points)
        mu_end = np.mean(end_points)

        sig_start = np.std(start_points)
        sig_end = np.std(end_points)


        ax.plot(start_points[:,0], start_points[:,1], "bo")
        # ax.plot(connected_end_points[:,0], connected_end_points[:,1], "g-")
        ax.plot(end_points[:,0], end_points[:,1], "go")
        # print(f"sig_end: {sig_end}")

        # estimator = EllipticEnvelope(support_fraction=1.0, contamination=0.25)
        # estimator.fit(end_points)

        # DecisionBoundaryDisplay.from_estimator(
        #     estimator,
        #     end_points,
        #     response_method="decision_function",
        #     plot_method="contour",
        #     levels=[0],
        #     colors="green",
        #     ax=ax,
        # )


h0 = np.load("h0.npy")
# h2 = np.load("h2.npy")

h0_pca = PCA(n_components=2)
# h2_pca = PCA(n_components=2)

# all_hidden_concat = np.concatenate(accumulated_all_hidden )
# print(f"{all_hidden_concat.shape}")
h0 = np.expand_dims(h0, axis=0)
print(f"h0: {h0.shape}")
h0_reshaped = h0.reshape(-1, h0.shape[-1])
print(f"h0_reshaped: {h0_reshaped.shape}")
h0_pca.fit(h0_reshaped)

# h2 = np.expand_dims(h2, axis=0)
# print(f"h2: {h2.shape}")
# h2_reshaped = h2.reshape(-1, h2.shape[-1])
# print(f"h2_reshaped: {h2_reshaped.shape}")
# h2_pca.fit(h2_reshaped)


# h2_reshaped = h2.reshape(-1, h2.shape[-1])
# h2_pca.fit(h2_reshaped)


# t = np.linspace(0, 10, 5000)
# print(t[0], t[-1], len(t))

# plt.plot(t, h0[:,0])
# plt.plot(t, h2[:,0])

fig, ax = plt.subplots()


# args={"title": "h0"}
# plot_hidden_trajectory(ax=ax, h=h0[:,:500,:], pca=h0_pca, args=args)
# h0_comp = h0_pca.components_

args={"title": "h0"}
plot_hidden_trajectory(ax=ax, h=h0[:,500:,:], pca=h0_pca, args=args)
h0_comp = h0_pca.components_




# args={"title": "h2"}
# plot_hidden_trajectory(ax=ax, h=h2[:,:1002,:], pca=h2_pca, args=args)
# h2_comp = h2_pca.components_

# args={"title": "h2"}
# plot_hidden_trajectory(ax=ax, h=h2[:,1002:,:], pca=h2_pca, args=args)
# h2_comp = h2_pca.components_

# h0_1000 = h0_comp @ h0_reshaped[1000]
# print(f"h0_1000: {h0_1000}")

# h0_1000 = h0_comp @ h0[0][1000]
# print(f"h0_1000: {h0_1000}")
# ax.plot(h0_1000[0], h0_1000[1], "ko")

# for i in range(h0.shape[1]):
#     h0_1000 = h0_comp @ h0[0][i]
#     ax.plot(h0_1000[0], h0_1000[1], "ko")
# for i in range(500):
#     h0_1000 = h0_comp @ h0[0][i]
#     ax.plot(h0_1000[0], h0_1000[1], "bx")
# for i in range(500, h0.shape[1]-1, 1):
#     h0_1000 = h0_comp @ h0[0][i]
#     ax.plot(h0_1000[0], h0_1000[1], "rx")

# for i in range(1002, 5000):
#     h0_1000 = h0_comp @ h0[0][i]
#     # print(f"h0_1000: {h0_1000}")
#     ax.plot(h0_1000[0], h0_1000[1], "yo")


# for i in range(1002):
#     h2_1000 = h2_comp @ h2[0][i]
#     # print(f"h0_1000: {h0_1000}")
#     ax.plot(h2_1000[0], h2_1000[1], "ko")

# for i in range(1002, 5000):
#     h2_1000 = h2_comp @ h2[0][i]
#     # print(f"h0_1000: {h0_1000}")
#     ax.plot(h2_1000[0], h2_1000[1], "yo")

# for i in range(1002, 5000):
#     h2_1000 = h2_comp @ h2[0][i]
#     # print(f"h0_1000: {h0_1000}")
#     ax.plot(h2_1000[0], h2_1000[1], "yo")


plt.show()