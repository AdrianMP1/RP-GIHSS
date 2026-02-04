import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tickeras ticker

from matplotlib import rc
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation
rc("text", usetex=True)


def lame_curve_generator(p: float, n_points: int, oversample: int = 1e9) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    oversample = int(1e6)

    # Generate dense points
    t_dense = np.linspace(0, np.pi/2, oversample)
    x_dense = np.cos(t_dense) ** (2/p)
    y_dense = np.sin(t_dense) ** (2/p)

    # Compute euclidean distance
    distances = np.sqrt(np.diff(x_dense)**2 + np.diff(y_dense)**2)
    cumulative_distances = np.cumsum(distances)
    cumulative_distances = np.insert(cumulative_distances, 0, 0)

    # Compute target distances including extremes
    total_distance = cumulative_distances[-1]
    target_distances = np.linspace(0, total_distance, n_points)

    # Find the indices in the dense data that corresponds to target distances
    indices = []
    current_indx = 0
    previous_error = float("inf")
    for target in target_distances:
        # Compute errors
        error = np.abs(cumulative_distances - target)

        # Get the one with minimum error
        index = np.argmin(errors)
        indices.append(index)

    # Ensure the extremes
    indices[0] = 0
    indices[-1] = len(x_dense) - 1

    # Extract the selected points
    x_uniform = x_dense[indices]
    y_uniform = y_dense[indices]

    return x_uniform, y_uniform


def compute_front(gamma: float, n_points: int = 100):
    x, y = lame_curve_generator(p=gamma, num_points=n_points)
    front = np.vstack([x,y]).T

    # Reverse to start at x=0.
    front = front[::-1]
    front[0] = np.round(front[0])
    return front


def setup_axes():
    # Low dpi
    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=120)
    ax.axis("scaled")

    # Ticks
    ax.tick_params(which="major", width=1.00, length=5)
    ax.tick_params(which="minor", width=0.75, length=2.5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    tick_values = [0, 0.5, 1.0]
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_values))
    ax.yaxis.set_major_locator(ticker.FixedLocator(tick_values))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.1f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.1f}"))

    # Labels
    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$")

    # Frame lines
    ax.axvline(x=0, lw=0.5, linestyle=":", c="k")
    ax.axhline(y=0, lw=0.5, linestyle=":", c="k")
    ax.axvline(x=1, lw=0.5, linestyle=":", c="k")
    ax.axhline(y=1, lw=0.5, linestyle=":", c="k")

    return fig, ax


def animate_alternating_pairs(concave_fronts, convex_fronts, special, steps_per_curve=60):
    # Create figure
    fig, ax = setup_axes()

    # Initial guide plot-lines
    bg_lines = []
    for p in special:
        x_bg, y_bg = lame_curve_interpolation_v2(p=p, num_points=100)
        bg_line, = ax.plot(x_bg, y_bg, lw=1.25, ls="--", c="k", zorder=0)
        bg_lines.append(bg_line)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()["color"]
    colors = cycle + cycle

    n_pairs = min(len(concave_fronts), len(convex_fronts))

    # One Line2D artist per curve
    conc_lines = []
    conv_lines = []
    for i in range(n_pairs):
        lc, = ax.plot([], [], lw=1.0, color="red")#colors[(2*i) % len(colors)])
        lv, = ax.plot([], [], lw=1.0, color="blue")#colors[(2*i+1) % len(colors)])
        conc_lines.append(lc)
        conv_lines.append(lv)

    # Frames
    frames_per_pair = 2 * steps_per_curve
    n_frames = n_pairs * frames_per_pair

    def init():
        for ln in conc_lines + conv_lines:
            ln.set_data([],[])
        #title.set_text("")
        return [*bg_lines, *conv_lines, *conv_lines]

    def set_full(i):
        fc = concave_fronts[i]
        fv = convex_fronts[i]
        conc_lines[i].set_data(fc[:, 0], fc[:, 1])
        conv_lines[i].set_data(fv[:, 0], fv[:, 1])

    def set_partial(line, front, k):
        # Map k
        nvis = int(2 + (len(front)-2) * (k / (steps_per_curve - 1)))
        line.set_data(front[:nvis, 0], front[:nvis, 1])

    def update(frame):
        pair_idx = frame // frames_per_pair
        r = frame % frames_per_pair
        phase = 0 if r < steps_per_curve else 1 # 0=concave, 1=convex
        k = r if phase == 0 else (r - steps_per_curve)

        # 1) all previous pairs fully drawn
        for i in range(pair_idx):
            set_full(i)

        # 2) current pair
        if phase == 0:
            # Reset current pair lines
            conc_lines[pair_idx].set_data([],[])
            conv_lines[pair_idx].set_data([],[])
            set_partial(conc_lines[pair_idx], concave_fronts[pair_idx], k)
            #title.set_text(f"pair {pair_idx+1}/{n_pairs} - concave")
        else:
            # Concave already complete, animate convex
            fc = concave_fronts[pair_idx]
            conc_lines[pair_idx].set_data(fc[:,0], fc[:,1])
            set_partial(conv_lines[pair_idx], convex_fronts[pair_idx], k)
        return [*bg_lines, *conc_lines, *conv_lines]

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=30, blit=True)
    return anim


# Run
# --- GAMMA Construction ---
convex_values = [0.1 + 0.05*i for i in range(18)]
concave_values = sorted([1/i for i in convex_values])

# Special gammas (dashed and black)
special = {1.0, 2.0, 1/2.0}

# Resolution per line
n_points = 100

# Make curves
convex_fronts = [compute_front(g, n_points) for g in convex_values]
concave_fronts = [compute_front(g, n_points) for g in concave_values]

# Animate
anim = animate_alternating_pairs(concave_fronts, convex_fronts[::-1], special, steps_per_curve=15)
HTML(anim.to_html5_video())

anim.save(
    "LAME_Curves.mp4",
    writer="ffmpeg",
    fps=30,
    dpi=120,
    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "veryfast"],
)
