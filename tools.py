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

    def display_frame(self, index, color_channel=None, v_min = None, v_max = None, x_up=None, x_down=None, y_left=None, y_right=None):
        """
        Display a specific frame using matplotlib.
        """
        frame = self.get_frame(index, color_channel)
        plt.figure(figsize=(8, 6))
        plt.imshow(frame, cmap='gray', aspect='auto', vmin=v_min, vmax=v_max)
        plt.xlim(y_left, y_right)
        plt.ylim(x_down, x_up)
        plt.title(f"Frame {index}")
        #plt.axis('off')
        plt.show()

    def release(self):
        self.cap.release()

def calculate_spatial_frequencies(frames, pixel_size, min_wave_number):
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
        freq_x = np.fft.fftfreq(width, d=1) # Horizontal frequencies
        freq_y = np.fft.fftfreq(height, d=1)  # Vertical frequencies

        # Convert to real-world units (cycles per unit distance)
        real_freq_x = freq_x / pixel_size
        real_freq_y = freq_y / pixel_size

        # Perform Fourier transform
        fft_result = fft2(frame)
        magnitude = np.abs(fft_result)
        # Average across axes for horizontal and vertical components
        horizontal_freq = np.array(magnitude[0,:])
        vertical_freq = np.array(magnitude[:,0])
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

def calculate_wavenumbers_1D(frames, pixel_size,detrend=False):
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
    for i,frame in tqdm.tqdm(enumerate(frames), desc="Processing frames"):
        if detrend:
            horizontal_avg = np.mean(frame, axis=0)
            frame = frame - horizontal_avg
            vertical_avg = np.mean(frame, axis=1)
            frame = frame - vertical_avg[:, np.newaxis]

        ### now set all outliers above 3 std to 0
        std = np.std(frame)
        frame[frame > 3*std] = 3*std
        frame[frame < -3*std] = -3*std

        if i == 0 or i == len(frames)-1:
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

    return wavenumbers, magnitudes



def fourier_analysis(video_loader, pixel_size, x_up=0, x_down=-1, y_left=0, y_right=-1,start_frame=0, end_frame=100,detrend=False):
    frames = []
    # Precompute slicing indices
    x_slice = slice(x_up, x_down)
    y_slice = slice(y_left, y_right)

    # Use list comprehension to load frames
    frames = [video_loader.get_frame(i)[x_slice, y_slice] for i in tqdm.tqdm(range(start_frame, end_frame))]
    frequencies, magnitudes = calculate_wavenumbers_1D(frames, pixel_size,detrend=detrend)
    ### average n values in each folder of freqencies
    return frequencies, magnitudes


def fourier_animation(frequencies, magnitudes,baseline_horizontal=0,baseline_vertical=0, average=20):
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
            np.mean(normalized_magnitudes[i:i+average], axis=0) 
            for i in range(0, len(normalized_magnitudes), average)
        ])
        
        # Average the frequencies as usual
        frequencies[key] = np.array([
            np.mean(frequencies[key][i:i+average], axis=0) 
            for i in range(0, len(frequencies[key]), average)
        ])
    baseline_vertical = baseline_vertical / baseline_vertical.max()
    baseline_horizontal = baseline_horizontal / baseline_horizontal.max()
    
    precomputed_differences = {
        "horizontal_diff": magnitudes["horizontal"] - baseline_horizontal,
        "vertical_diff": magnitudes["vertical"] - baseline_vertical,
        "horizontal_freq": frequencies["horizontal"],
        "vertical_freq":  frequencies["vertical"]
    }
    def update(frame_index):
        diff_h = precomputed_differences["horizontal_diff"][frame_index]
        diff_v = precomputed_differences["vertical_diff"][frame_index]
        freq_h = precomputed_differences["horizontal_freq"][frame_index]
        freq_v = precomputed_differences["vertical_freq"][frame_index]
        #print(diff_h, diff_v, freq_h, freq_v)
        # Update scatter plots
        horizontal_scatter.set_offsets(np.c_[freq_h, diff_h])
        vertical_scatter.set_offsets(np.c_[freq_v, diff_v])
        # ax[0].set_title(f"Horizontal Fourier Transform (average of frames {frame_index * average}-{frame_index * average + average})")
        return horizontal_scatter, vertical_scatter

    # Create animations for horizontal and vertical frequencies
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    # Scatter plot placeholders
    def init_func():
        ax[0].set_xlim(0,frequencies["horizontal"].max())
        ax[0].set_ylim(precomputed_differences["horizontal_diff"].min()*0.9,
                        precomputed_differences["horizontal_diff"].max()*1.1)
        ax[0].set_yscale('symlog', linthresh=0.1)  # Symmetric log scale for differences
        ax[0].set_title("Horizontal Fourier Transform")
        ax[0].set_xlabel("Wavenumber k (1/mm)")
        ax[0].set_ylabel("Magnitude")
        ax[0].legend()
        ax[0].grid()

        ax[1].set_xlim(0,frequencies["vertical"].max())
        ax[1].set_ylim(precomputed_differences["vertical_diff"].min()*0.9,
                       precomputed_differences["vertical_diff"].max()*1.1)
        ax[1].set_yscale('symlog', linthresh=0.1)  # Symmetric log scale for differences
        ax[1].set_title("Vertical Fourier Transform")
        ax[1].set_xlabel("Wavenumber k (1/mm)")
        ax[1].set_ylabel("Magnitude")
        ax[1].legend()
        ax[1].grid()

    horizontal_scatter = ax[0].scatter([], [], label="Horizontal Difference", color="blue", s=1)
    vertical_scatter = ax[1].scatter([], [], label="Vertical Difference", color="red", s=1)
    ani = FuncAnimation(fig, update, frames=len(precomputed_differences["horizontal_diff"]), init_func=init_func, blit=False, repeat=True)
    # plt.show()
    return ani