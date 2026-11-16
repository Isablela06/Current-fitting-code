import pybdsim
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

#=======================================
# User Input
#=======================================

#root file name to be analysed
INPUT_FILE = "1x3_127_Combine_Hires_Histos.root"

#name of the output directory for plots and csv
OUTPUT_DIR = Path("DepthDosePlots")

#prefix for the mesh names and the total number of meshes
#assumes the naming convention MESHNAME1, MESHNAME2, ... MESHNAME_NMESHES
MESH_NAMES = "WaterBackMesh"
N_MESHES = 13

#number of beams in each direction
#must be odd and at least 3, with the central beam at x = 0, y = 0
#for example, 3 beams in x and 5 in y gives a 3x5 grid centred on (0, 0)
NUM_BEAMS_X = 3
NUM_BEAMS_Y = 3

#beam spacing in mm, assumes equal spacing in x and y
BEAM_SPACING = 3

#scaling factor for dose
DOSE_SCALE = 1e10

#width of the area used to average dose at each valley, in mm
VALLEY_AVERAGE_WIDTH = 0.5

#initial guess for the gaussian sigma, in mm
INITIAL_SIGMA = 0.5

#=======================================
# Fit Equation
#=======================================

def fit_function(coordinates, a, sigma):
    """
    Sum NUM_BEAMS_X * NUM_BEAMS_Y gaussians with the same amplitude and sigma

    The beam centres are fixed on a uniform grid centred at (0, 0)
    """

    #converts the coordinates to arrays and broadcasts them to the same shape
    x, y = np.broadcast_arrays(
        np.asarray(coordinates[0], dtype=float),
        np.asarray(coordinates[1], dtype=float),
    )

    #starts the result array with zeros at each coordinate
    result = np.zeros_like(x)

    #adds the gaussian from each beam to the dose at each coordinate
    for j in range(NUM_BEAMS_Y):
        for i in range(NUM_BEAMS_X):
            mux = (i - (NUM_BEAMS_X - 1) / 2) * BEAM_SPACING
            muy = (j - (NUM_BEAMS_Y - 1) / 2) * BEAM_SPACING
            result += a * np.exp(
                -0.5 * (((x - mux) / sigma) ** 2 + ((y - muy) / sigma) ** 2)
            )

    #returns a flat array for scipy curve_fit
    return result.ravel()

#=======================================
# Analysis Functions
#=======================================

def find_local_average(x_grid, y_grid, dose, centre_x, centre_y, window_width):
    #finds the average dose inside a square window around the given point
    mask = (
        (np.abs(x_grid - centre_x) <= window_width / 2)
        & (np.abs(y_grid - centre_y) <= window_width / 2)
    )

    #uses the nearest bin if the window does not contain any bin centres
    if not np.any(mask):
        distance_squared = (x_grid - centre_x) ** 2 + (y_grid - centre_y) ** 2
        index = np.unravel_index(np.argmin(distance_squared), dose.shape)
        return float(dose[index])

    #returns the average dose inside the window
    return float(np.mean(dose[mask]))


def find_metrics(x_grid, y_grid, dose, sigma):
    #stores the positions and dose averages for each peak and valley
    peak_positions = []
    peak_averages = []
    valley_positions = []
    valley_averages = []

    #finds the average dose around the centre of each beam
    for j in range(NUM_BEAMS_Y):
        for i in range(NUM_BEAMS_X):
            #gets the position of the current beam centre
            centre_x = (i - (NUM_BEAMS_X - 1) / 2) * BEAM_SPACING
            centre_y = (j - (NUM_BEAMS_Y - 1) / 2) * BEAM_SPACING
            peak_positions.append((centre_x, centre_y))

            #uses a small area around the beam centre to find its peak dose
            local_average = find_local_average(
                x_grid,
                y_grid,
                dose,
                centre_x=centre_x,
                centre_y=centre_y,
                window_width=0.1 * abs(sigma),
            )
            peak_averages.append(local_average)

    #finds the average dose where four neighbouring beams meet
    for j in range(NUM_BEAMS_Y - 1):
        for i in range(NUM_BEAMS_X - 1):
            #gets the position halfway between each pair of neighbouring beams
            valley_x = (i - (NUM_BEAMS_X - 2) / 2) * BEAM_SPACING
            valley_y = (j - (NUM_BEAMS_Y - 2) / 2) * BEAM_SPACING
            valley_positions.append((valley_x, valley_y))

            #averages dose over a small area around the four-beam valley
            local_average = find_local_average(
                x_grid,
                y_grid,
                dose,
                centre_x=valley_x,
                centre_y=valley_y,
                window_width=VALLEY_AVERAGE_WIDTH,
            )
            valley_averages.append(local_average)

    #converts the lists to arrays for the calculations below
    peak_positions = np.asarray(peak_positions, dtype=float)
    peak_averages = np.asarray(peak_averages)
    valley_positions = np.asarray(valley_positions, dtype=float)
    valley_averages = np.asarray(valley_averages)

    #calculates the average dose across all peaks and valleys
    mean_peak = np.mean(peak_averages)
    mean_valley = np.mean(valley_averages)

    #calculates the peak-to-valley dose ratio if the valley dose is non-zero
    pvdr = mean_peak / mean_valley if mean_valley > 0 else np.nan

    #keeps undefined or infinite ratios out of the results
    if not np.isfinite(pvdr):
        pvdr = np.nan

    #returns the positions, averages and overall pvdr
    return {
        "peak_positions": peak_positions,
        "peak_averages": peak_averages,
        "valley_positions": valley_positions,
        "valley_averages": valley_averages,
        "mean_peak": mean_peak,
        "mean_valley": mean_valley,
        "PVDR": pvdr
    }

#=======================================
# Main Function
#=======================================

#creates the output directory if it does not already exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#loads the data from the input ROOT file
data = pybdsim.Data.Load(INPUT_FILE)
#gets the 3D histograms from the data object
histograms = data.histograms3dpy

#initialises lists and arrays to store the results for each mesh
results = []
reduced_chi_squared_arr = np.array([])
fitted_PVDR_arr = np.array([])
raw_PVDR_arr = np.array([])
depth_dose_profiles = []
depth_y_positions = None

#loops through each mesh and fits it as a 2D beam grid
for mesh_number in range(1, N_MESHES + 1):
    #adds the current mesh number to the mesh name prefix
    mesh_name = MESH_NAMES + str(mesh_number)

    #gets the histogram for the current mesh
    h = histograms[f'Event/MergedHistograms/{mesh_name}-ProtonInteraction']

    #gets the dose and errors from the first z slice of the (x, y, z) histogram
    dose = np.asarray(h.contents[:, :, 0], dtype=float)
    errors = np.asarray(h.errors[:, :, 0], dtype=float)

    #gets the x and y bin centres and converts them from metres to millimetres
    x = np.asarray(h.xcentres, dtype=float) * 1000
    y = np.asarray(h.ycentres, dtype=float) * 1000

    #uses the y positions from the first mesh for the depth plot
    if depth_y_positions is None:
        depth_y_positions = y.copy()
    elif not np.allclose(y, depth_y_positions):
        raise ValueError(f"{mesh_name} has different y bin centres from the other meshes")

    #makes a grid of x and y positions to match the dose array
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    #scales the dose values and their errors
    dose *= DOSE_SCALE
    errors *= DOSE_SCALE

    #finds bins with finite, positive errors to use in the weighted fit
    finite_errors = np.isfinite(errors)
    positive_errors = finite_errors & (errors > 0)

    #stops if no bins have a usable error
    if not np.any(positive_errors):
        raise ValueError(f"{mesh_name} has no positive finite uncertainties")

    #removes errors so small that they cannot be used reliably in the fit
    uncertainty_cutoff = np.finfo(float).eps * np.max(errors[positive_errors])

    #keeps only bins with finite dose and usable errors
    valid_bins = np.isfinite(dose) & positive_errors & (errors > uncertainty_cutoff)

    #stops if there are not enough usable bins to fit the two parameters
    if np.count_nonzero(valid_bins) <= 2:
        raise ValueError(f"{mesh_name} has too few bins with valid uncertainties to fit")

    #scales the dose values to keep the fitting calculations well-conditioned
    dose_scale = float(np.max(np.abs(dose[valid_bins])))

    #stops if there is no non-zero dose to fit
    if dose_scale == 0:
        raise ValueError(f"{mesh_name} has no non-zero dose values to fit")

    #sets the initial guesses for the shared amplitude and sigma
    initial_guess = [float(np.max(dose[valid_bins])) / dose_scale, INITIAL_SIGMA]

    #fits the dose grid with one shared amplitude and sigma
    #the dose and errors are both scaled to keep the fit stable
    scaled_parameters, pcov = curve_fit(
        fit_function,
        (x_grid[valid_bins], y_grid[valid_bins]),
        dose[valid_bins] / dose_scale,
        p0=initial_guess,
        sigma=errors[valid_bins] / dose_scale,
        absolute_sigma=True,
        bounds=([0, np.finfo(float).eps], [np.inf, np.inf]),
        x_scale="jac",
        maxfev=20000,
    )

    #converts the fitted amplitude back to the original dose scale
    popt = scaled_parameters.copy()
    popt[0] *= dose_scale

    #calculates the uncertainties in the fitted parameters
    parameter_errors = np.sqrt(np.diag(pcov)) * np.array([dose_scale, 1])

    #calculates the metrics from the original dose grid
    raw_metrics = find_metrics(x_grid, y_grid, dose, popt[1])

    #calculates the fitted dose values at the original grid positions
    fitted_data = fit_function(
        (x_grid.ravel(), y_grid.ravel()), *popt
    ).reshape(dose.shape)

    #calculates the metrics from the fitted dose grid
    fitted_metrics = find_metrics(x_grid, y_grid, fitted_data, popt[1])

    #calculates the residuals between the original and fitted dose values
    residuals = dose - fitted_data

    #calculates the chi-squared and reduced chi-squared using the usable bins
    chi_squared = np.sum((residuals[valid_bins] / errors[valid_bins]) ** 2)
    degrees_of_freedom = np.count_nonzero(valid_bins) - len(popt)
    reduced_chi_squared = chi_squared / degrees_of_freedom

    #stores the dose profile at the x bin closest to zero for the depth plot
    central_x_index = int(np.argmin(np.abs(x)))
    depth_dose_profiles.append(dose[central_x_index, :].copy())

    # =======================================
    # plots the original dose and fitted surface
    # =======================================

    #makes a finer grid to show the fit as a smooth surface
    x_dense = np.linspace(x.min(), x.max(), 120)
    y_dense = np.linspace(y.min(), y.max(), 120)
    x_dense_grid, y_dense_grid = np.meshgrid(x_dense, y_dense, indexing="ij")

    #calculates the fit at each point on the finer grid
    fitted_data_dense = fit_function(
        (x_dense_grid.ravel(), y_dense_grid.ravel()), *popt
    ).reshape(x_dense_grid.shape)

    #sets up a 3D plot for the dose surfaces
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    #plots the original dose with colour showing its value
    original_surface = ax.plot_surface(
        x_grid,
        y_grid,
        dose,
        cmap="viridis",
        alpha=0.8,
        linewidth=0,
        antialiased=True,
        label="Original BDSIM dose",
    )

    #plots the fitted gaussian surface over the original dose
    ax.plot_surface(
        x_dense_grid,
        y_dense_grid,
        fitted_data_dense,
        color="darkorange",
        alpha=0.5,
        linewidth=0,
        antialiased=True,
        label="Constrained Gaussian fit",
    )

    #marks the raw and fitted peak and valley averages on the plot
    for metrics, color, marker, suffix in (
        (raw_metrics, "black", "^", "raw"),
        (fitted_metrics, "red", "v", "fit"),
    ):
        #gets the peak and valley positions for the current set of metrics
        peak_positions = metrics["peak_positions"]
        valley_positions = metrics["valley_positions"]

        #plots the peak averages at the beam centres
        ax.scatter(
            peak_positions[:, 0],
            peak_positions[:, 1],
            metrics["peak_averages"],
            color=color,
            marker=marker,
            s=35,
            label=f"Peak averages ({suffix})",
        )

        #plots the valley averages where four beams meet
        ax.scatter(
            valley_positions[:, 0],
            valley_positions[:, 1],
            metrics["valley_averages"],
            color=color,
            marker="x",
            s=35,
            label=f"Four-beam valley averages ({suffix})",
        )

    #adds a colour scale for the original dose values
    fig.colorbar(original_surface, ax=ax, pad=0.1, label="Dose per second / Gy s⁻¹")

    #adds the fit quality and pvdr values to the plot title
    ax.set_title(
        f"{mesh_name}: 2D Fixed-Centre Gaussian Fit\n"
        f"Raw PVDR = {raw_metrics['PVDR']:.4g}, "
        f"Fitted PVDR = {fitted_metrics['PVDR']:.4g}\n"
        f"reduced χ² = {reduced_chi_squared:.3g}"
    )

    #labels the position and dose axes
    ax.set_xlabel("Transverse displacement x / mm")
    ax.set_ylabel("Transverse displacement y / mm")
    ax.set_zlabel("Dose per second / Gy s⁻¹")

    #adds the legend and grid, then saves and closes the plot
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{mesh_name}.pdf")
    plt.close(fig)

    # =======================================
    # plots the original dose and fitted points in 2D
    # =======================================

    #sets up a 2D plot of the original dose grid
    fig, ax = plt.subplots(figsize=(10, 8))

    #plots the original dose with colour showing its value
    dose_image = ax.pcolormesh(
        x,
        y,
        dose.T,
        shading="auto",
        cmap="viridis",
    )

    #plots the fitted peak positions over the original dose
    ax.scatter(
        fitted_metrics["peak_positions"][:, 0],
        fitted_metrics["peak_positions"][:, 1],
        marker="^",
        color="magenta",
        edgecolors="black",
        s=55,
        label="Fitted beam centres",
    )

    #plots the fitted four-beam valley positions over the original dose
    ax.scatter(
        fitted_metrics["valley_positions"][:, 0],
        fitted_metrics["valley_positions"][:, 1],
        marker="x",
        color="white",
        linewidths=2,
        s=55,
        label="Fitted four-beam valleys",
    )

    #adds a colour scale and labels the position axes
    fig.colorbar(dose_image, ax=ax, label="Dose per second / Gy s⁻¹")
    ax.set_title(f"{mesh_name}: Original Dose and Fitted Points")
    ax.set_xlabel("Transverse displacement x / mm")
    ax.set_ylabel("Transverse displacement y / mm")
    ax.set_aspect("equal", adjustable="box")

    #adds the legend and grid, then saves and closes the plot
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{mesh_name}_Dose2D.pdf")
    plt.close(fig)

    # =======================================
    # stores the results for later plots and csv output
    # =======================================

    #adds the fit metrics to the summary arrays
    reduced_chi_squared_arr = np.append(reduced_chi_squared_arr, reduced_chi_squared)
    raw_PVDR_arr = np.append(raw_PVDR_arr, raw_metrics['PVDR'])
    fitted_PVDR_arr = np.append(fitted_PVDR_arr, fitted_metrics['PVDR'])

    #stores the peak and valley metrics for this mesh
    results.append({
        "peak_positions_raw": raw_metrics["peak_positions"],
        "peak_averages_raw": raw_metrics["peak_averages"],
        "valley_positions_raw": raw_metrics["valley_positions"],
        "valley_averages_raw": raw_metrics["valley_averages"],
        "mean_peak_raw": raw_metrics["mean_peak"],
        "mean_valley_raw": raw_metrics["mean_valley"],
        "PVDR_raw": raw_metrics["PVDR"],
        "peak_positions_fitted": fitted_metrics["peak_positions"],
        "peak_averages_fitted": fitted_metrics["peak_averages"],
        "valley_positions_fitted": fitted_metrics["valley_positions"],
        "valley_averages_fitted": fitted_metrics["valley_averages"],
        "mean_peak_fitted": fitted_metrics["mean_peak"],
        "mean_valley_fitted": fitted_metrics["mean_valley"],
        "PVDR_fitted": fitted_metrics["PVDR"],
        "reduced_chi_squared": reduced_chi_squared,
    })


#=======================================
# Further Analysis
#=======================================

#plots the dose at x = 0 against y position and mesh number
mesh_numbers = np.arange(1, len(depth_dose_profiles) + 1)
depth_dose_grid = np.asarray(depth_dose_profiles, dtype=float)

#sets up a heatmap to show how the dose changes through the sample
fig, ax = plt.subplots(figsize=(12, 8))
depth_image = ax.pcolormesh(
    mesh_numbers,
    depth_y_positions,
    np.ma.masked_invalid(depth_dose_grid.T),
    shading="nearest",
    cmap="viridis",
)

#adds a colour scale and labels the depth and y position axes
fig.colorbar(depth_image, ax=ax, label="Dose per second / Gy s⁻¹")
ax.set_xlim(0.5, len(mesh_numbers) + 0.5)
ax.set_xticks(mesh_numbers)
ax.set_xlabel("Depth into sample / mesh number")
ax.set_ylabel("Transverse displacement y / mm")
ax.set_title("Dose vs Depth and y Displacement at x = 0")

#adds a grid, then saves and closes the plot
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "Dose_vs_Depth_and_Y.pdf")
plt.close(fig)

#plots the raw and fitted pvdr values for each mesh
fig, ax = plt.subplots(figsize=(12, 7))

#ignores non-finite and non-positive pvdr values before taking the logarithm
fitted_pvdr_mask = np.isfinite(fitted_PVDR_arr) & (fitted_PVDR_arr > 0)
raw_pvdr_mask = np.isfinite(raw_PVDR_arr) & (raw_PVDR_arr > 0)

#plots pvdr on a log10 scale for the raw and fitted values
ax.errorbar(
    np.flatnonzero(fitted_pvdr_mask),
    np.log10(fitted_PVDR_arr[fitted_pvdr_mask]),
    fmt="o",
    color="red",
    markersize=7,
    capsize=3,
    label="Fitted PVDR",
)
ax.errorbar(
    np.flatnonzero(raw_pvdr_mask),
    np.log10(raw_PVDR_arr[raw_pvdr_mask]),
    fmt="s",
    color="blue",
    markersize=7,
    capsize=3,
    label="Raw PVDR",
)

#labels the plot and sets one tick for each mesh
ax.set_xlabel("Mesh Number")
ax.set_ylabel("log₁₀(PVDR)")
ax.set_title("PVDR vs Mesh Number")
ax.set_xticks(range(len(fitted_PVDR_arr)))

#adds the legend and grid, then saves and closes the plot
ax.legend()
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "PVDR_vs_Mesh.pdf")
plt.close(fig)

#plots the reduced chi-squared value for each mesh
fig, ax = plt.subplots(figsize=(12, 7))

#plots one point for each mesh
ax.errorbar(
    range(len(reduced_chi_squared_arr)),
    reduced_chi_squared_arr,
    fmt="o",
    color="red",
    markersize=7,
    capsize=3,
    label="reduced χ²",
)

#labels the plot and sets one tick for each mesh
ax.set_xlabel("Mesh Number")
ax.set_ylabel("reduced χ²")
ax.set_title("reduced χ² vs Mesh Number")
ax.set_xticks(range(len(reduced_chi_squared_arr)))

#adds the legend and grid, then saves and closes the plot
ax.legend()
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "RCS_vs_Mesh.pdf")
plt.close(fig)

#converts the results to a dataframe and saves them to a csv file
df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "results.csv", index=False)