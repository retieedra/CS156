# Feel free to skip over this cell, it's just plotting code.
def plot_decision_surface(clf, xs, ys, ax, title):
    xs = np.linspace(xs[0], xs[1], 100)
    ys = np.linspace(ys[0], ys[1], 100)
    x_mesh, y_mesh = np.meshgrid(xs, ys)

    vis_x = np.stack((x_mesh, y_mesh), axis=2).reshape(-1, 2)
    vis_y = clf.predict_proba(vis_x)[:, 1].reshape(100, 100)

    contour = ax.contourf(
        x_mesh, y_mesh, vis_y, levels=np.linspace(0, 1, 33), cmap="Blues"
    )

    contour.set_clim(0, 1)
    ax.scatter(
        x_data[:, 0],
        x_data[:, 1],
        c=y_data,
        edgecolors="grey",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
