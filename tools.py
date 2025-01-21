import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, fftshift
import tqdm
from IPython.display import HTML
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import gsw
import xarray as xr


class AVIFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open file: {file_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_number_of_frames(self):
        return self.total_frames

    def get_frame(self, index, color_channel=None):
        if index < 0 or index >= self.total_frames:
            raise IndexError(f"Index {index} is out of range. Total frames: {self.total_frames}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame at index {index}")

        if color_channel is not None:
            # Extract the specified color channel (0=Blue, 1=Green, 2=Red)
            return frame[:, :, color_channel].astype(np.float32)
        else:
            # Default to grayscale
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def display_frame(self, index, color_channel=None, v_min=None, v_max=None, x_up=None, x_down=None, y_left=None,
                      y_right=None):
        """
        Display a specific frame using matplotlib.
        """
        frame = self.get_frame(index, color_channel)
        plt.figure(figsize=(8, 6))
        plt.imshow(frame, cmap='gray', aspect='auto', vmin=v_min, vmax=v_max)
        plt.xlim(y_left, y_right)
        plt.ylim(x_down, x_up)
        plt.title(f"Frame {index}")
        # plt.axis('off')
        plt.show()

    def release(self):
        self.cap.release()


def find_local_minima(lst, window_size):
    result = []
    indices = []
    n = len(lst)
    for i in range(n):
        left = max(0, i - window_size)
        right = min(n, i + window_size)
        if lst[i] == min(lst[left:right]) and (
                len(indices) >= 1 and indices[-1] + window_size <= i or len(indices) == 0):
            result.append(lst[i])
            indices.append(i)
    return result, indices


def find_gridsize(frame, x_up=0, x_down=-1, y_left=0, y_right=-1, window_size=30):
    frame = frame[x_up:x_down, y_left:y_right]
    plt.figure()
    ### add x and y labels
    plt.imshow(frame, cmap='gray', aspect='auto')
    slice = int(frame.shape[0] / 2)
    plt.hlines(slice, 0, len(frame[slice]), color='r', alpha=0.2)
    plt.title('Frame')
    plt.ylabel('x')
    plt.xlabel('y')
    ### change y and x ticks to between x_up and x_down and y_left and y_right
    plt.xticks(np.arange(0, frame.shape[1], step=int(frame.shape[1] / 5)),
               np.arange(y_left, y_right, step=int(frame.shape[1] / 5)))
    plt.yticks(np.arange(0, frame.shape[0], step=int(frame.shape[0] / 5)),
               np.arange(x_up, x_down, step=int(frame.shape[0] / 5)))
    plt.show()
    minima, i_min = find_local_minima(frame[slice], window_size)
    plt.plot(range(len(frame[slice])), frame[slice])

    for i in i_min:
        plt.plot(i, frame[slice][i], 'ro')
    plt.show()
    print(np.diff(i_min))
    return np.mean(np.diff(i_min))


def find_size(loader, x_up=0, x_down=-1, y_left=0, y_right=-1, color_channel=None):
    n_fr = loader.get_number_of_frames()
    frame = loader.get_frame(int(n_fr / 2), color_channel)
    loader.display_frame(int(n_fr / 2), color_channel)

    window_frame = frame[x_up:x_down, y_left:y_right]
    plt.imshow(window_frame, cmap='gray', aspect='auto')
    plt.xticks(np.arange(0, window_frame.shape[1], step=int(window_frame.shape[1] / 8)),
               np.arange(y_left, y_right, step=int(window_frame.shape[1] / 8)))
    plt.yticks(np.arange(0, window_frame.shape[0], step=int(window_frame.shape[0] / 8)),
               np.arange(x_up, x_down, step=int(window_frame.shape[0] / 8)))
    return x_up, x_down, y_left, y_right


def calculate_wavenumbers_1D(video_loader, start_frame, end_frame, x_slice, y_slice, pixel_size, detrend=False):
    """
    Calculate wavenumbers (spatial frequencies) for a list of frames with scaling to real-world units.

    Args:
        frames (list of np.ndarray): List of grayscale frames.
        pixel_size (float): Size of each pixel in real-world units (e.g., mm or µm).

    Returns:
        dict: Contains horizontal and vertical wavenumbers and magnitudes for all frames.
            - wavenumbers["horizontal"]: Average horizontal wavenumbers (1/unit distance).
            - wavenumbers["vertical"]: Vertical wavenumbers (1/unit distance).
            - magnitudes["horizontal"]: Average horizontal magnitude spectrum for each frame.
            - magnitudes["vertical"]: Average vertical magnitude spectrum for each frame.
    """
    # Initialize lists to store results
    wn_x_l = []  # Horizontal wavenumbers
    wn_y_l = []  # Vertical wavenumbers
    horizontal_mag_l = []  # Horizontal magnitudes
    vertical_mag_l = []  # Vertical magnitudes

    # Process each frame
    for i in tqdm.tqdm(range(start_frame, end_frame + 1), desc="Processing frames"):
        # Load the frame
        frame = video_loader.get_frame(i)[x_slice, y_slice]
        if detrend:
            frame = detrend_and_del_outliers(frame)

        if i == start_frame or i == end_frame:
            c = plt.imshow(frame, cmap='coolwarm', aspect='auto')
            plt.colorbar(c)
            plt.title(f"Frame {i}")
            plt.show()

        # Frame dimensions
        height, width = frame.shape

        # Wavenumbers in pixel units
        wn_x = np.fft.fftfreq(width, d=1) / pixel_size  # Horizontal wavenumbers
        wn_y = np.fft.fftfreq(height, d=1) / pixel_size  # Vertical wavenumbers

        # Horizontal wavenumbers: Fourier analysis for each row, then average
        horizontal_magnitude = np.mean(np.abs(np.fft.fft(frame, axis=1)), axis=0)

        # Vertical wavenumbers: Fourier analysis for each column, then average
        vertical_magnitude = np.mean(np.abs(np.fft.fft(frame, axis=0)), axis=1)

        # Append results for this frame
        wn_x_l.append(wn_x)
        wn_y_l.append(wn_y)
        horizontal_mag_l.append(horizontal_magnitude)
        vertical_mag_l.append(vertical_magnitude)

    # Compile results into dictionaries
    wavenumbers = {
        "horizontal": np.array(wn_x_l),
        "vertical": np.array(wn_y_l)}
    magnitudes = {
        "horizontal": np.array(horizontal_mag_l),
        "vertical": np.array(vertical_mag_l)}

    #### return only the positive wavenumbers values
    horizontal_mask = wavenumbers["horizontal"][0, :] > 0
    vertical_mask = wavenumbers["vertical"][0, :] > 0

    # Apply the mask to the correct axis
    wavenumbers["horizontal"] = wavenumbers["horizontal"][:, horizontal_mask]
    magnitudes["horizontal"] = magnitudes["horizontal"][:, horizontal_mask]
    wavenumbers["vertical"] = wavenumbers["vertical"][:, vertical_mask]
    magnitudes["vertical"] = magnitudes["vertical"][:, vertical_mask]

    return wavenumbers, magnitudes


def fourier_analysis(video_loader, pixel_size, x_up=0, x_down=-1, y_left=0, y_right=-1, start_frame=0, end_frame=100,
                     detrend=False):
    frames = []
    # Precompute slicing indices
    x_slice = slice(x_up, x_down)
    y_slice = slice(y_left, y_right)

    # Use list comprehension to load frames
    frequencies, magnitudes = calculate_wavenumbers_1D(video_loader, start_frame, end_frame, x_slice, y_slice,
                                                       pixel_size, detrend=detrend)
    ### average n values in each folder of freqencies
    return frequencies, magnitudes


def gaussian_fit(x, y, x_min=0, x_max=0.5):
    """
    Fit a Gaussian curve to the given data points.

    Args:
        x (np.ndarray): Independent variable values.
        y (np.ndarray): Dependent variable values.

    Returns:
        tuple: Parameters of the Gaussian curve (amplitude, mean, standard deviation).
    """
    # Filter data points within the specified range
    mask = (x >= x_min) & (x <= x_max)
    x = x[mask]
    y = y[mask]

    from scipy.optimize import curve_fit

    # Define the Gaussian function
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    try:
        # Perform curve fitting
        popt, _ = curve_fit(gaussian, x, y, p0=[1, np.mean(x), np.std(x)], maxfev=10000)

        return popt  # Return the fit parameters (amplitude, mean, standard deviation)
    except (RuntimeError, ValueError) as e:
        # If fitting fails, print the error message and return None
        print(f"Gaussian fit failed: {e}")
        return None


# Define the power decay function
def power_decay(x, A, n, B):
    return A / (x ** n) + B


def Power_Decay_Fit(x, y):
    """
    Fit a power decay curve to the given data points.

    Args:
        x (np.ndarray): Independent variable values.
        y (np.ndarray): Dependent variable values.

    Returns:
        tuple: Parameters of the power decay curve (amplitude, decay rate).
    """
    from scipy.optimize import curve_fit
    # Filter data points within the specified range
    mask = (x > 0)
    x = x[mask]
    y = y[mask]

    try:
        # Perform curve fitting
        popt, _ = curve_fit(power_decay, x, y, p0=[1, 1, 1], maxfev=10000)

        return popt  # Return the fit parameters (amplitude, decay rate)
    except (RuntimeError, ValueError) as e:
        # If fitting fails, print the error message and return None
        print(f"Power decay fit failed: {e}")
        return None


def fourier_animation(frequencies, magnitudes, baseline_horizontal=0, baseline_vertical=0, average=20, fit_min=0.1,
                      fit_max=0.5, n_max_mag=10):
    frequencies = frequencies.copy()
    magnitudes = magnitudes.copy()
    for key in frequencies.keys():
        # Normalize magnitudes for each frame by the max magnitude of that frame
        normalized_magnitudes = [
            frame_magnitude / frame_magnitude.max() if frame_magnitude.max() != 0 else frame_magnitude
            for frame_magnitude in magnitudes[key]
        ]

        # Replace magnitudes[key] with normalized values
        magnitudes[key] = np.array([
            np.mean(normalized_magnitudes[i:i + average], axis=0)
            for i in range(0, len(normalized_magnitudes), average)
        ])

        # Average the frequencies as usual
        frequencies[key] = np.array([
            np.mean(frequencies[key][i:i + average], axis=0)
            for i in range(0, len(frequencies[key]), average)
        ])

    if type(baseline_horizontal) != int:
        baseline_vertical = baseline_vertical / baseline_vertical.max()
        baseline_horizontal = baseline_horizontal / baseline_horizontal.max()
    else:
        ### perform fit for the magnitudes
        baseline_horizontal = []
        for i in range(len(magnitudes["vertical"])):
            fit_params = Power_Decay_Fit(frequencies["horizontal"][i], magnitudes["horizontal"][i])
            baseline_h = power_decay(frequencies["horizontal"][i], fit_params[0], fit_params[1], fit_params[2])
            baseline_horizontal.append(baseline_h)
            plt.scatter(frequencies["horizontal"][i], magnitudes["horizontal"][i])
            plt.plot(frequencies["horizontal"][i], baseline_h)
            plt.show()

    precomputed_differences = {
        "horizontal_diff": magnitudes["horizontal"] - baseline_horizontal,
        "vertical_diff": magnitudes["vertical"] - baseline_vertical,
        "horizontal_freq": frequencies["horizontal"],
        "vertical_freq": frequencies["vertical"]
    }

    def update(frame_index):
        diff_h = precomputed_differences["horizontal_diff"][frame_index]
        diff_v = precomputed_differences["vertical_diff"][frame_index]
        freq_h = precomputed_differences["horizontal_freq"][frame_index]
        freq_v = precomputed_differences["vertical_freq"][frame_index]

        ### give the wavenumber values of the n_max_mag highest magnitude values and take only into account wave number between fit_min and fit_max
        mask_wn = (freq_h > fit_min) & (freq_h < fit_max)
        mask_max_mag = np.argsort(diff_h[mask_wn])[::-1][:n_max_mag]
        saltfingers_wn = freq_h[mask_wn][mask_max_mag]

        ### calculate weighted average of the wavenumbers
        weighted_avg = np.average(saltfingers_wn, weights=diff_h[mask_wn][mask_max_mag])

        # Perform Gaussian fit for the horizontal and vertical differences
        horizontal_fit_params = gaussian_fit(frequencies["horizontal"][frame_index],
                                             precomputed_differences["horizontal_diff"][frame_index], fit_min, fit_max)
        # Update scatter plots
        horizontal_scatter.set_offsets(np.c_[freq_h, diff_h])
        vertical_scatter.set_offsets(np.c_[freq_v, diff_v])
        # Update vertical line for weighted average
        vertical_line.set_xdata([weighted_avg, weighted_avg])
        vertical_line.set_label(
            f"Weighted Average (wavenumber): {weighted_avg:.3f} wavelenght (mm): {1 / weighted_avg:.3f}")
        ## plot only if the fit is successful
        if horizontal_fit_params is not None:
            horizontal_gaussian_line.set_label(
                f"Amplitude: {horizontal_fit_params[0]:.3f}, Mean: {horizontal_fit_params[1]:.3f}, Std: {horizontal_fit_params[2]:.3f}, Wavelength (mm) {1 / horizontal_fit_params[1]:.3f}")
            horizontal_gaussian_line.set_data(frequencies["horizontal"][frame_index], horizontal_fit_params[0] * np.exp(
                -0.5 * ((frequencies["horizontal"][frame_index] - horizontal_fit_params[1]) / horizontal_fit_params[
                    2]) ** 2))
        ### add label to the line plot indicating the fit parameters
        ax[0].legend()
        return horizontal_scatter, vertical_scatter, horizontal_gaussian_line

    # Create the animation figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Scatter plot placeholders
    def init_func():
        ax[0].set_xlim(0, frequencies["horizontal"].max())
        ax[0].set_ylim(-0.1,
                       precomputed_differences["horizontal_diff"].max() * 1.1)
        # ax[0].set_xscale('symlog', linthresh=0.1)  # Symmetric log scale for differences
        ax[0].set_title("Horizontal Fourier Transform")
        ax[0].set_xlabel("Wavenumber k (1/mm)")
        ax[0].set_ylabel("Magnitude")
        ax[0].legend()
        ax[0].grid()

        ax[1].set_xlim(0, frequencies["vertical"].max())
        ax[1].set_ylim(precomputed_differences["vertical_diff"].min() * 0.9,
                       precomputed_differences["vertical_diff"].max() * 1.1)
        # ax[1].set_xscale('symlog', linthresh=0.1)  # Symmetric log scale for differences
        ax[1].set_title("Vertical Fourier Transform")
        ax[1].set_xlabel("Wavenumber k (1/mm)")
        ax[1].set_ylabel("Magnitude")
        ax[1].legend()
        ax[1].grid()

    # Scatter plots for horizontal and vertical differences
    horizontal_scatter = ax[0].scatter([], [], label="Horizontal Difference", color="blue", s=1)
    vertical_line = ax[0].axvline(x=0, color='r', linestyle='--', label='Weighted Average')
    vertical_scatter = ax[1].scatter([], [], label="Vertical Difference", color="red", s=1)

    # Lines for the Gaussian fits
    horizontal_gaussian_line, = ax[0].plot([], [], color="green", linestyle="--")
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(precomputed_differences["horizontal_diff"]), init_func=init_func,
                        blit=False, repeat=True)

    return ani


import ipywidgets as widgets
from ipywidgets import interact


def display_baseline_selector(video_loader, par_window):
    """
    Create an interactive widget to select and confirm the baseline frame.

    Args:
        video_loader: Object with a `display_frame` method for showing video frames.
        par_window: Dictionary defining the display window (x_up, x_down, y_left, y_right).

    Returns:
        The confirmed baseline frame number.
    """
    # Variable to store the confirmed baseline frame
    confirmed_frame = {"baseline": None}

    def find_baseline(frame_num, v_min, v_max):
        """
        Display a specific video frame with defined v_min and v_max values.
        """
        video_loader.display_frame(
            frame_num,
            x_up=par_window['x_up'],
            x_down=par_window['x_down'],
            y_left=par_window['y_left'],
            y_right=par_window['y_right'],
            v_min=v_min,
            v_max=v_max,
        )
        print(f"Displaying frame {frame_num} with v_min={v_min}, v_max={v_max}")

    def on_button_click(b):
        """
        Handle the button click to confirm the baseline frame.
        """
        confirmed_frame["baseline"] = frame_slider.value
        print(f"Baseline frame confirmed: {confirmed_frame['baseline']}")

    # Create interactive widgets
    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=500,
        step=1,
        description='Frame number:',
    )

    v_min = widgets.IntSlider(
        value=0,
        min=0,
        max=255,
        step=1,
        description='v_min:',
    )

    v_max = widgets.IntSlider(
        value=255,
        min=0,
        max=255,
        step=1,
        description='v_max:',
    )

    # Create a button to confirm the baseline frame
    button = widgets.Button(description="Confirm Baseline Frame")
    button.on_click(on_button_click)

    # Display the widgets
    display(widgets.VBox([frame_slider, v_min, v_max, button]))
    interact(find_baseline, frame_num=frame_slider, v_min=v_min, v_max=v_max)

    # Return the confirmed frame number
    return confirmed_frame


def cube(x, m, a, b):
    return m * x ** 2 + a * x + b


def cube_fit(x, y):
    """
    Fit a linear curve to the given data points.

    Args:
        x (np.ndarray): Independent variable values.
        y (np.ndarray): Dependent variable values.

    Returns:
        tuple: Parameters of the linear curve (slope, intercept).
    """
    from scipy.optimize import curve_fit

    try:
        # Perform curve fitting
        popt, _ = curve_fit(cube, x, y)

        return popt  # Return the fit parameters (slope, intercept)
    except (RuntimeError, ValueError) as e:
        # If fitting fails, print the error message and return None
        print(f"Linear fit failed: {e}")
        return None


def detrend_and_del_outliers(frame):
    # detrend every row of the frame with linear fit
    detrended_frame = np.zeros_like(frame)
    for i in range(frame.shape[0]):
        popt = cube_fit(np.arange(frame.shape[1]), frame[i])
        detrended_frame[i] = frame[i] - cube(np.arange(frame.shape[1]), *popt)
    # normalize the frame by subtracting the mean an then divide by the std
    detrended_frame = detrended_frame - np.mean(detrended_frame)
    detrended_frame = detrended_frame / np.std(detrended_frame)
    # delete all values above 3 std
    std = np.std(detrended_frame)
    detrended_frame[detrended_frame > 3 * std] = 3 * std
    detrended_frame[detrended_frame < -3 * std] = -3 * std

    return detrended_frame


def find_start_and_end_frame(video_loader, par_window, start_frame_num, end_frame_num):
    """
    Plot the start and end frames after detrending and removing outliers.

    Args:
        video_loader: Object with a `get_frame` method for fetching frames.
        par_window: Dictionary defining the display window (x_up, x_down, y_left, y_right).
        start_frame_num: Frame number for the start frame.
        end_frame_num: Frame number for the end frame.
    """
    x_slice = slice(par_window['x_up'], par_window['x_down'])
    y_slice = slice(par_window['y_left'], par_window['y_right'])

    start_frame = video_loader.get_frame(start_frame_num)[x_slice, y_slice]
    start_frame = detrend_and_del_outliers(start_frame)
    end_frame = video_loader.get_frame(end_frame_num)[x_slice, y_slice]
    end_frame = detrend_and_del_outliers(end_frame)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    c_s = ax[0].imshow(start_frame, cmap='coolwarm', aspect='auto')
    c_e = ax[1].imshow(end_frame, cmap='coolwarm', aspect='auto')
    ax[0].set_title("Start Frame")
    ax[1].set_title("End Frame")
    plt.colorbar(c_s, ax=ax[0])
    plt.colorbar(c_e, ax=ax[1])
    plt.show()


def interactive_start_end_selector(video_loader, par_window):
    """
    Create an interactive widget to select and confirm start and end frames.

    Args:
        video_loader: Object with a `get_frame` method for fetching frames.
        par_window: Dictionary defining the display window (x_up, x_down, y_left, y_right).

    Returns:
        A tuple (start_frame_num, end_frame_num) containing the confirmed frame numbers.
    """
    # Variables to store confirmed start and end frames
    confirmed_frames = {"start": None, "end": None}

    # Create sliders for start and end frame selection
    start_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=video_loader.get_number_of_frames() - 1,
        step=1,
        description="Start Frame:",
    )
    end_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=video_loader.get_number_of_frames() - 1,
        step=1,
        description="End Frame:",
    )

    # Create buttons to confirm start and end frames
    start_button = widgets.Button(description="Confirm Start Frame")
    end_button = widgets.Button(description="Confirm End Frame")

    def confirm_start(b):
        confirmed_frames["start"] = start_slider.value
        print(f"Start frame confirmed: {confirmed_frames['start']}")

    def confirm_end(b):
        confirmed_frames["end"] = end_slider.value
        print(f"End frame confirmed: {confirmed_frames['end']}")

    # Attach button callbacks
    start_button.on_click(confirm_start)
    end_button.on_click(confirm_end)

    # Create the interactive function for displaying frames
    @interact
    def update_display(start_frame=start_slider, end_frame=end_slider):
        find_start_and_end_frame(video_loader, par_window, start_frame, end_frame)

    # Display widgets
    display(widgets.VBox([start_slider, start_button, end_slider, end_button]))

    # Wait for user confirmation
    return confirmed_frames


class TemperatureProcessor:
    def __init__(self, file_path, z_positions=None):
        if z_positions is None:
            z_positions = {
                '03': np.arange(0, 15, 2),  # Stick 03: z = 0 to 14
                '02': np.arange(1, 16, 2),  # Stick 02: z = 1 to 15
                '01': np.arange(7.5, 22.5, 2),  # Stick 01: z = 10.5 to 24.5
                '00': np.arange(8.5, 23.5, 2)  # Stick 00: z = 11.5 to 25.5
            }
        self.file_path = file_path
        self.data = None
        self.dataset = None
        self.z_positions = z_positions

    def read_data(self):
        """Reads and cleans the temperature data."""
        columns = ['Time', 'Stick', 'ID', 'Timestamp'] + [f"Temp_{i}" for i in range(8)]
        # Explicitly set Stick column as string during reading
        self.data = pd.read_csv(
            self.file_path,
            sep=',|\t',
            engine='python',
            names=columns,
            dtype={'Stick': str}
        )

        # Ensure Stick column has leading zeros (if accidentally stripped)
        self.data['Stick'] = self.data['Stick'].apply(lambda x: x.zfill(2))

        # Keep only required columns
        self.data = self.data[['Time', 'Stick'] + [f"Temp_{i}" for i in range(8)]]
        offsets = pd.read_csv("offset_matrix.csv")
        #offsets["0"] =


    def process_data_to_xarray(self):
        """Processes the data and stores it as an xarray.Dataset."""
        time_groups = self.data[self.data['Stick'].isin(['00', '01', '02', '03'])].groupby('Time')

        time_coords = []
        stick_coords = ['00', '01', '02', '03']
        z_coords = {stick: self.z_positions[stick] for stick in stick_coords}
        temp_data = {stick: [] for stick in stick_coords}

        for time, group in time_groups:
            time_coords.append(time)
            group = group.set_index('Stick')
            for stick in stick_coords:
                if stick in group.index:
                    # Extract temperatures and pad to ensure correct shape
                    temps = group.loc[stick, [f"Temp_{i}" for i in range(8)]].values.astype(float)
                    temp_data[stick].append(temps)
                else:
                    # Pad with NaNs if stick data is missing for this timestamp
                    temp_data[stick].append([np.nan] * 8)

        # Convert to xarray Dataset
        data_vars = {}
        for stick in stick_coords:
            data_vars[f"temperature_{stick}"] = (
                ['time', 'z'],
                np.array(temp_data[stick]),
                {'description': f"Temperatures for stick {stick}"}
            )

        self.dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': time_coords,
                'z_00': z_coords['00'],
                'z_01': z_coords['01'],
                'z_02': z_coords['02'],
                'z_03': z_coords['03']
            },
            attrs={'description': 'Temperature time series processed into xarray Dataset'}
        )
        time = self.dataset.time
        t0 = self.dataset.temperature_00
        z0 = self.dataset.z_00
        t1 = self.dataset.temperature_01
        z1 = self.dataset.z_01
        t2 = self.dataset.temperature_02
        z2 = self.dataset.z_02
        t3 = self.dataset.temperature_03
        z3 = self.dataset.z_03
        temp = np.concatenate([t0, t1, t2, t3], axis=1)
        z = np.concatenate([z0, z1, z2, z3], axis=0)
        t_sticks = xr.Dataset(
            data_vars={
                "temperature": (["time", "z"], temp)  # Correctly specify the variable name and dimensions
            },
            coords={
                'time': time.values,  # Ensure time is in a compatible format (like np.array or list)
                'z': z  # z-coordinate values
            },
            attrs={
                'description': 'Temperature time series processed into xarray Dataset'
            }
        )
        ### sort the dataset in z directions
        t_sticks = t_sticks.sortby("z")
        # Fill NaN values with the temperature at the same z from the last timestep where there was a value
        t_sticks['temperature'] = t_sticks['temperature'].ffill(dim='time')
        self.dataset = t_sticks

    def get_dataset(self):
        """Returns the xarray.Dataset."""
        return self.dataset


def r_roh(SA_1, SA_2,CT_1, CT_2,dz,p):
    SA_mean = (SA_1 + SA_2) / 2
    CT_mean = (CT_1 + CT_2) / 2
    SA_dz = (SA_2 - SA_1) / dz
    CT_dz = (CT_2 - CT_1) / dz
    beta = gsw.beta(SA_mean, CT_mean, p)
    alpha = gsw.alpha(SA_mean, CT_mean, p)
    return alpha*CT_dz / (beta*SA_dz)


def fastes_growing_k(SA_1, SA_2,CT_1, CT_2,dz,p):
    SA_mean = (SA_1 + SA_2) / 2
    CT_mean = (CT_1 + CT_2) / 2
    SA_dz = (SA_2 - SA_1) / dz
    beta = gsw.beta(SA_mean, CT_mean, p)
    r = r_roh(SA_1, SA_2,CT_1, CT_2, dz, p)
    print("R_rho: ", r)
    k = 1.4e-7  # m^2/s thermal diffusivity
    v = 10e-6  # m^2/s kinematic viscosity
    g = 9.81  # m/s^2

    return np.sqrt(np.sqrt(g*beta*SA_dz*(r - 1)/(k*v)))
