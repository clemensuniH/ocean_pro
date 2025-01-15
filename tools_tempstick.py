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

