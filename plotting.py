# Import necessary libraries
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the .cdf file
file_path = 'data/3.cdf'
dataset = xr.open_dataset(file_path)

# Extract temperature data
temperature = dataset['temp'].values
time = dataset['Time'].values

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize the plot with the first frame
cax = ax.imshow(temperature[0, :, 0, :], cmap='viridis')
fig.colorbar(cax, label='Temperature')
ax.set_title('Temperature Data Over Time')
ax.set_xlabel('x')
ax.set_ylabel('z')

# Update function for animation
def update(frame):
    cax.set_data(temperature[frame, :, 0, :])
    ax.set_title(f'Temperature Data at Time {time[frame]:.4f}')
    return cax, ax

# Create the animation
ani = FuncAnimation(fig, update, frames=len(time), blit=False)

# Display the animation
plt.show()