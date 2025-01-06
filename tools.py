import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, fftshift
import tqdm
from IPython.display import HTML


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
            return frame[:, :, color_channel]
        else:
            # Default to grayscale
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def display_frame(self, index, color_channel=None, v_min = None, v_max = None, xlim_l = None, xlim_u = None, ylim_l = None, ylim_u = None):
        """
        Display a specific frame using matplotlib.
        """
        frame = self.get_frame(index, color_channel)
        plt.figure(figsize=(8, 6))
        plt.imshow(frame, cmap='gray', aspect='auto', vmin=v_min, vmax=v_max)
        plt.ylim(ylim_u, ylim_l)
        plt.xlim(xlim_l, xlim_u)
        plt.title(f"Frame {index}")
        #plt.axis('off')
        plt.show()

    def release(self):
        self.cap.release()

def calculate_spatial_frequencies(frames, pixel_size):
    """
    Calculate spatial frequencies for a list of frames with scaling to real-world units.

    Args:
        frames (list of np.ndarray): List of grayscale frames.
        pixel_size (float): Size of each pixel in real-world units (e.g., mm or µm).

    Returns:
        list of tuples: Each tuple contains horizontal and vertical frequencies for a frame.
    """
    real_freq_x_l = []
    real_freq_y_l = []
    horizontal_mag_l = []
    vertical_mag_l = []
    frequencies = {}
    magnitudes = {}
    for frame in tqdm.tqdm(frames):
        # Frame dimensions
        height, width = frame.shape

        # Spatial frequencies in pixel units
        freq_x = np.fft.fftfreq(width, d=1)  # Horizontal frequencies
        freq_y = np.fft.fftfreq(height, d=1)  # Vertical frequencies

        # Convert to real-world units (cycles per unit distance)
        real_freq_x = freq_x / pixel_size
        real_freq_y = freq_y / pixel_size

        # Perform Fourier transform
        fft_result = fft2(frame)
        magnitude = np.abs(fft_result)
        # Average across axes for horizontal and vertical components
        horizontal_freq = magnitude[0,:]
        vertical_freq = magnitude[:,0]
        real_freq_x_l.append(real_freq_x)
        real_freq_y_l.append(real_freq_y)
        horizontal_mag_l.append(horizontal_freq)
        vertical_mag_l.append(vertical_freq)
    
    frequencies["horizontal"] = np.array(real_freq_x_l)
    frequencies["vertical"] = np.array(real_freq_y_l)
    magnitudes["horizontal"] = np.array(horizontal_mag_l)
    magnitudes["vertical"] = np.array(vertical_mag_l)
    return frequencies, magnitudes


def find_local_minima(lst, window_size):
    result = []
    indices = []
    n = len(lst)
    for i in range(n):
        left = max(0, i - window_size)
        right = min(n, i + window_size)
        if lst[i] == min(lst[left:right]) and (len(indices) >=1 and indices[-1] + window_size <= i or len(indices) == 0):
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
    plt.xticks(np.arange(0,frame.shape[1],step=int(frame.shape[1]/5)), np.arange(y_left,y_right,step=int(frame.shape[1]/5)))
    plt.yticks(np.arange(0,frame.shape[0],step=int(frame.shape[0]/5)), np.arange(x_up,x_down,step=int(frame.shape[0]/5)))
    plt.show()
    minima, i_min = find_local_minima(frame[slice], window_size)
    plt.plot(range(len(frame[slice])), frame[slice])

    for i in i_min:
        plt.plot(i, frame[slice][i], 'ro')
    plt.show()
    print(np.diff(i_min))
    return np.mean(np.diff(i_min))


def find_size(loader, x_up=0, x_down=-1, y_left=0, y_right=-1,color_channel=None):
    n_fr = loader.get_number_of_frames()
    frame = loader.get_frame(int(n_fr/2),color_channel)
    loader.display_frame(int(n_fr/2),color_channel)
    #whole_frame = frame[0:-1, 0:-1]
    #plt.xticks(np.arange(0,whole_frame.shape[1],step=int(whole_frame.shape[1]/15)), np.arange(0,whole_frame.shape[1],step=int(whole_frame.shape[1]/15)))
    #plt.yticks(0,whole_frame.shape[0])
    window_frame = frame[x_up:x_down, y_left:y_right]
    plt.imshow(window_frame, cmap='gray', aspect='auto')
    plt.xticks(np.arange(0,window_frame.shape[1],step=int(window_frame.shape[1]/8)), np.arange(y_left,y_right,step=int(window_frame.shape[1]/8)))
    plt.yticks(np.arange(0,window_frame.shape[0],step=int(window_frame.shape[0]/8)), np.arange(x_up,x_down,step=int(window_frame.shape[0]/8)))
    return x_up, x_down, y_left, y_right

def fourier_analysis(video_loader, resolution, x_up=0, x_down=-1, y_left=0, y_right=-1, baseline_horizontal=0,baseline_vertical=0,start_frame=0, end_frame=100, average=20):
    frames = []
    for i in tqdm.tqdm(range(start_frame,end_frame)): 
        frame = video_loader.get_frame(i)
        frames.append(frame[x_up:x_down, y_left:y_right])
        
    frequencies, magnitudes = calculate_spatial_frequencies(frames, resolution)
    ### average n values in each folder of freqencies
    for key in frequencies.keys():
    #plt.plot(frequencies[key], magnitudes[key], label=key)
        frequencies[key] = np.array([np.mean(frequencies[key][i:i+20],axis=0) for i in range(0, len(frequencies[key]), average)])
        magnitudes[key] = np.array([np.mean(magnitudes[key][i:i+20],axis=0) for i in range(0, len(magnitudes[key]), average)])

    
    precomputed_differences = {
        "horizontal_diff": np.abs(magnitudes["horizontal"] - baseline_horizontal),
        "vertical_diff": np.abs(magnitudes["vertical"] - baseline_vertical),
        "horizontal_freq": frequencies["horizontal"],
        "vertical_freq":  frequencies["vertical"]
    }
    return precomputed_differences


def fourier_animation(precomputed_differences):
    def update(frame_index):
        diff_h = precomputed_differences["horizontal_diff"][frame_index]
        diff_v = precomputed_differences["vertical_diff"][frame_index]
        freq_h = precomputed_differences["horizontal_freq"][frame_index]
        freq_v = precomputed_differences["vertical_freq"][frame_index]
        #print(diff_h, diff_v, freq_h, freq_v)
        # Update scatter plots
        horizontal_scatter.set_offsets(np.c_[freq_h, diff_h])
        vertical_scatter.set_offsets(np.c_[freq_v, diff_v])

        return horizontal_scatter, vertical_scatter

    # Create animations for horizontal and vertical frequencies
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    # Scatter plot placeholders
    def init_func():
        # ax[0].set_xlim(0,1)
        # ax[0].set_ylim(1, 1e8)
        ax[0].set_yscale('symlog', linthresh=0.1)  # Symmetric log scale for differences
        ax[0].set_title("Horizontal Fourier Transform")
        ax[0].set_xlabel("Freq (1/mm)")
        ax[0].set_ylabel("Magnitude")
        ax[0].legend()
        ax[0].grid()

        # ax[1].set_xlim(0, 1)
        # ax[1].set_ylim(1, 1e8)
        ax[1].set_yscale('symlog', linthresh=0.1)  # Symmetric log scale for differences
        ax[1].set_title("Vertical Fourier Transform")
        ax[1].set_xlabel("Freq (1/mm)")
        ax[1].set_ylabel("Magnitude")
        ax[1].legend()
        ax[1].grid()

    horizontal_scatter = ax[0].scatter([], [], label="Horizontal Difference", color="blue", s=1)
    vertical_scatter = ax[1].scatter([], [], label="Vertical Difference", color="red", s=1)
    ax[1].set_xlim(0, 2)
    ax[0].set_xlim(0, 2)
    ax[0].set_ylim(precomputed_differences["horizontal_diff"].min()-1, precomputed_differences["horizontal_diff"].max()+1)
    ax[1].set_ylim(precomputed_differences["vertical_diff"].min()-1, precomputed_differences["vertical_diff"].max()+1)
    ani = FuncAnimation(fig, update, frames=len(precomputed_differences["horizontal_diff"]), init_func=init_func, blit=False, repeat=True)
    #plt.show()
    return ani
    



import pandas as pd
import xarray as xr

class TemperatureProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.dataset = None
        self.z_positions = {
            '03': np.arange(0, 15, 2),      # Stick 03: z = 0 to 14
            '02': np.arange(1, 16, 2),      # Stick 02: z = 1 to 15
            '01': np.arange(9.5, 24.5, 2), # Stick 01: z = 10.5 to 24.5
            '00': np.arange(10.5, 25.5, 2)  # Stick 00: z = 11.5 to 25.5
        }

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

    def get_dataset(self):
        """Returns the xarray.Dataset."""
        return self.dataset

# Usage Example (Not implemented here):
# file_path = 'temp.txt'
# processor = TemperatureProcessor(file_path)
# processor.read_data()
# processor.process_data_to_xarray()
# dataset = processor.get_dataset()
# print(dataset)

