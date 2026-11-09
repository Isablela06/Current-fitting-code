import pybdsim
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

#=======================================
# User Input
#=======================================

#Root file name to be analysed
INPUT_FILE = "1x3_127_Combine_Hires_Histos.root"

#Name of output directory for plots and csv
OUTPUT_DIR = Path("DepthDosePlots")

#Prefix for meshes and the number of them assuming naming convention of
# MESHNAME1, MESHNAME2, MESHNAME3 ... MESHNAME_NMESHES
MESH_NAMES = "WaterBackMesh"
N_MESHES = 13

# Must be odd, must be positive, must be at least 3
# assumes central beam at X = 0, Y = 0 
# i.e. numbeams 3 gives a 1x3 grid of beams extending in +ve and -ve y direction
NUM_BEAMS_X = 3

# Beam spacing in mm, assumes uniform spacing in y dirn
BEAM_SPACING = 3

#Scaling factor for dose
DOSE_SCALE = 1e10

#Valley averaging function width in mm
VALLEY_AVERAGE_WIDTH = 0.5

#intial guess for the sigma parameter in the fit function, in mm
INITIAL_SIGMA = 0.5  # Initial guess for sigma in mm

#=======================================
# Fit Equation
#=======================================

def fit_function(x, a, sigma):
    """
    sum of N (N = NUM_BEAMS_X) gaussians with some paramaters constrained

    Fixed centers at x = 0 +- (NUM_BEAMS_X-1)/2 * BEAM_SPACING

    Sigma of all gaussians is the same

    Amplitudes of gaussians are constrained to be the same
    
    D(x) = sum_i A * exp(-(x-mu_i)^2 / (2*sigma^2))
    """

    #converting x to numpy array to ensure it works with scipy curve_fit
    x = np.asarray(x, dtype=float)

    #initialize result array with zeros of the same shape as x
    result = np.zeros_like(x)
    
    for i in range(NUM_BEAMS_X):
        mu = (i - (NUM_BEAMS_X - 1) / 2) * BEAM_SPACING
        
        result += a * np.exp(-(x - mu)**2 / (2 * sigma**2))

    return result

#single gaussian component for plotting individual gaussians in the fit
def gaussian_component(x, amplitude, centre, sigma):
    return amplitude * np.exp(-0.5 * ((x - centre) / sigma)**2)


#=======================================
# Recursive Functions
#=======================================

def find_local_average(x, dose, centre_x, window_width):
    #finds a local average around a point centre_x with a window of width window_width
    
    #makes an array of boolean values where True indicates the x value is within the window
    mask = np.abs(x - centre_x) <= window_width / 2

    #tests the edge case that the window width is smaller than the x bin widths
    #returns single closest dose to centre_x in this case
    if not np.any(mask):
        index = np.argmin(np.abs(x - centre_x))
        return float(dose[index])

    #returns the average of the dose values within the window
    return float(np.mean(dose[mask]))

def find_metrics(x, dose, sigma):
    #general analysis on the dose profile to find local averages at each beam center,
    #valley and calculates pvdr for those averages

    #initializes lists to hold the positions and averages of peaks and valleys
    peak_positions = []
    peak_averages = []
    valley_positions = []
    valley_averages = []

    #iterates over all beam centers and finds the local average dose at each center
    for i in range(NUM_BEAMS_X):
        #finds the x position of the current beam center
        mu = (i - (NUM_BEAMS_X - 1) / 2) * BEAM_SPACING
        peak_positions.append(mu)

        #calls for a local average around that beam center with a window width of 0.1sigma
        local_average = find_local_average(x, dose, centre_x=mu, window_width=0.1 * sigma)
        peak_averages.append(local_average)

    #iterates over the valleys between the beams to find the local average dose at each valley
    for i in range(NUM_BEAMS_X - 1):
        #finds the x position of the current valley, which is halfway between two beam centers
        mu1 = (i - (NUM_BEAMS_X - 1) / 2) * BEAM_SPACING
        mu2 = ((i + 1) - (NUM_BEAMS_X - 1) / 2) * BEAM_SPACING
        valley_x = (mu1 + mu2) / 2
        valley_positions.append(valley_x)

        #calls for a local average around that valley with a window width of VALLEY_AVERAGE_WIDTH
        local_average = find_local_average(x, dose, centre_x=valley_x, window_width=VALLEY_AVERAGE_WIDTH)
        valley_averages.append(local_average)

    #converts the lists to numpy arrays for easier mathematical operations
    peak_positions = np.asarray(peak_positions)
    peak_averages = np.asarray(peak_averages)
    valley_positions = np.asarray(valley_positions)
    valley_averages = np.asarray(valley_averages)

    #finds the mean of the peak and valley values
    mean_peak = np.mean(peak_averages)
    mean_valley = np.mean(valley_averages)

    #calculates the peak-to-valley dose ratio (PVDR) as the mean peak divided by the mean valley with exception for valley dose being zero to prevent div / 0 err
    pvdr = mean_peak / mean_valley if mean_valley > 0 else np.nan

    #returns a dictionary containing the calculated metrics for plotting and further analysis
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

#create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the data from the input ROOT file
data = pybdsim.Data.Load(INPUT_FILE)
#strips histograms from the data object for easier access
histograms = data.histograms3dpy

#initialises some arrays to hold results for analysis
results = []
reduced_chi_squared_arr = np.array([])
fitted_PVDR_arr = np.array([])
raw_PVDR_arr = np.array([])

#main loop to extract x profiles for each mesh
for mesh_number in range(1, N_MESHES + 1):
    #mesh_name is a concatination of the MESH_NAMES prefix and the current mesh_number inline
    #with naming convention of MESH_NAMES1, MESH_NAMES2, ..., MESH_NAMES_NMESHES
    mesh_name = MESH_NAMES + str(mesh_number)

    #strips out just the histogram we are looking for out of the list of all histograms
    h = histograms[f'Event/MergedHistograms/{mesh_name}-ProtonInteraction']

    #since each scoring mesh has only one bin in y and z we pull only a 1d 
    # array of dose values and errors for the x profile
    dose = np.asarray(h.contents[:, 0, 0], dtype=float)
    errors = np.asarray(h.errors[:, 0, 0], dtype=float)

    #get the x position of each bin center in mm, converting from m to mm
    x = np.asarray(h.xcentres, dtype=float) * 1000 

    dose *= DOSE_SCALE  # Scale the dose values
    errors *= DOSE_SCALE  # Scale the error values

    #finds an initial guess for a by averaging a tiny window around the central peak
    initial_a = find_local_average(x, dose, centre_x=0, window_width=0.01)

    #initial guess for the fit parameters, a and sigma
    initial_guess = [initial_a, INITIAL_SIGMA]

    #calls scipy curve_fit to fit the dose profile to the fit_function defined above
    popt, pcov = curve_fit(fit_function, x, dose, p0=initial_guess, sigma=errors, absolute_sigma=True, maxfev=20000)

    #extract the fitted parameters uncertainties
    parameter_errors = np.sqrt(np.diag(pcov))

    #populates the minimum and maximum bounds as the outermost beam centers plus 5 standard deviations of the beam width
    x_min = (-((NUM_BEAMS_X - 1) / 2) * BEAM_SPACING) - 5 * popt[1]
    x_max = (((NUM_BEAMS_X - 1) / 2) * BEAM_SPACING) + 5 * popt[1]

    #gathers the raw metrics from the original dose profile and the fitted metrics from the fitted dose profile
    raw_metrics = find_metrics(x, dose, popt[1])

    #gathers the fitted metrics from the fitted dose profile
    fitted_metrics = find_metrics(x, fit_function(x, *popt), popt[1])

    #calculates the fitted dose values and residuals for chi-squared calculation
    fitted_data = fit_function(x, *popt)
    residuals = dose - fitted_data

    #calculates the chi-squared value and reduced chi-squared value for the fit
    chi_squared = np.sum((residuals / errors) ** 2)
    degrees_of_freedom = len(x) - len(popt)
    reduced_chi_squared = chi_squared / degrees_of_freedom

    # =======================================
    # creates plots
    # =======================================

    #creates a densely sampled x array for plotting the fitted function smoothly
    x_dense = np.linspace(x_min, x_max, 4000)

    #finds the fitted dose values for the densely sampled x array
    fitted_data_dense = fit_function(x_dense, *popt)

    fig, ax = plt.subplots(figsize=(12, 7))

    #plots the original dose profile with error bars
    ax.errorbar(x, dose, yerr=errors, fmt='o', label='Original BDSIM Data', color='black', ecolor='grey', alpha=0.55, capsize=0, markersize=3)

    #plots the fitted function as a smooth line
    ax.plot(x_dense, fitted_data_dense, label='Constrained Fit', color='darkorange', linewidth=2.2)

    #plots the individual gaussian components of the fit as dotted lines
    for i in range(NUM_BEAMS_X):
        #calculates the individual gaussian component for each beam in the fit and plots it as a dotted line
        gauss_component = gaussian_component(x_dense, popt[0], (i - (NUM_BEAMS_X - 1) / 2) * BEAM_SPACING, popt[1])
        
        ax.plot(x_dense, gauss_component, linestyle=":", linewidth=1.5, color="black", label=f"Gaussian {i+1} (centre {(i - (NUM_BEAMS_X - 1) / 2) * BEAM_SPACING} mm)")

    #plots the mean peak and valley doses for both the raw data and the fitted data as horizontal dashed lines
    ax.axhline(raw_metrics["mean_peak"],color="tomato",linestyle="--",alpha=0.35,linewidth=1.6,label="Mean peak dose (raw)")
    ax.axhline(raw_metrics["mean_valley"],color="chartreuse",linestyle="--",alpha=0.35,linewidth=1.6,label="Mean valley dose (raw)")

    ax.axhline(fitted_metrics["mean_peak"],color="red",linestyle="--",alpha=0.35,linewidth=1.6,label="Mean peak dose (fit)")
    ax.axhline(fitted_metrics["mean_valley"],color="green",linestyle="--",alpha=0.35,linewidth=1.6,label="Mean valley dose (fit)")

    #plots the fitted peak and valley positions and doses for both the raw data and the fitted data as error bars with markers
    ax.errorbar(raw_metrics["peak_positions"],raw_metrics["peak_averages"],fmt="^",color="tomato",markersize=7,capsize=3,label="Fitted peaks (raw)")
    ax.errorbar(raw_metrics["valley_positions"],raw_metrics["valley_averages"],fmt="v",color="chartreuse",markersize=7,capsize=3,label="Fitted valleys (raw)")

    ax.errorbar(fitted_metrics["peak_positions"],fitted_metrics["peak_averages"],fmt="^",color="red",markersize=7,capsize=3,label="Fitted peaks (fit)")
    ax.errorbar(fitted_metrics["valley_positions"],fitted_metrics["valley_averages"],fmt="v",color="green",markersize=7,capsize=3,label="Fitted valleys (fit)")

    #finishes the plot with labels, title, legend, grid and saves it to the output directory
    ax.set_title(f"{mesh_name}: Fixed-Centre Gaussian Fit\n"f"Raw PVDR = {raw_metrics['PVDR']:.4g}, Fitted PVDR = {fitted_metrics['PVDR']:.4g}\n"f"reduced χ² = {reduced_chi_squared:.3g}")

    ax.set_xlabel("Transverse displacement x / mm")
    ax.set_ylabel("Dose per second / Gy s⁻¹")

    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / f"{mesh_name}.pdf")

    plt.close(fig)

    # =======================================
    # Data Cleanup
    # =======================================
    
    #saves raw PVDR, fitted PVDR and reduced chi squared to arrays for use in later plots
    reduced_chi_squared_arr = np.append(reduced_chi_squared_arr, reduced_chi_squared)
    raw_PVDR_arr = np.append(raw_PVDR_arr, raw_metrics['PVDR'])
    fitted_PVDR_arr = np.append(fitted_PVDR_arr, fitted_metrics['PVDR'])

    #saves all results to a list of dictionaries for later use in CSV
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
    "reduced_chi_squared": reduced_chi_squared})


# =======================================
# further analysis
# =======================================

#creates a plot of PVDR vs mesh number for all meshes, using the raw and fitted PVDR values 
fig, ax = plt.subplots(figsize=(12, 7))
ax.errorbar(range(len(fitted_PVDR_arr)), fitted_PVDR_arr, fmt="o", color="red", markersize=7, capsize=3, label="Fitted PVDR")
ax.errorbar(range(len(raw_PVDR_arr)), raw_PVDR_arr, fmt="s", color="blue", markersize=7, capsize=3, label="Raw PVDR")
ax.set_xlabel("Mesh Number")
ax.set_ylabel("log(PVDR)")
ax.set_title("PVDR vs Mesh Number")
ax.set_xticks(range(len(fitted_PVDR_arr)))
plt.yscale("log")
ax.legend()
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "PVDR_vs_Mesh.pdf")
plt.close(fig)

#creates a plot of reduced chi squared vs mesh number for all meshes
fig, ax = plt.subplots(figsize=(12, 7))
ax.errorbar(range(len(reduced_chi_squared_arr)), reduced_chi_squared_arr, fmt="o", color="red", markersize=7, capsize=3, label="reduced χ²")
ax.set_xlabel("Mesh Number")
ax.set_ylabel("reduced χ²")
ax.set_title("reduced χ² vs Mesh Number")
ax.set_xticks(range(len(reduced_chi_squared_arr)))
ax.legend()
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "RCS_vs_Mesh.pdf")
plt.close(fig)

#saves all the information in results to a CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / f"results.csv", index=False)