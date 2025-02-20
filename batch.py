from logger_setup import logger
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

class Batch:
    """
    Class representing a batch of devices.
    """

    def __init__(self,                     # initialise Batch class
                 batch_name: str,           # name of batch
                 batch_dir: str,           # relevant directories start with this string
                 substrates_file: str,     # path to substratesfile (excel)
                 config_file: str          # path to config file (excel)    
                 ):

        # initialise batch directory
        self.batch_name = batch_name
        self.batch_dir = batch_dir
        logger.info(f'Initialising batch from {self.batch_dir}')

        self.hrs = self._get_measurement_hours()
        self.substrates = pd.read_csv(substrates_file)
        self.config = pd.read_excel(config_file)
        self.data = self.gen_df()

    def _get_measurement_hours(self) -> list:
    # extracts unique measurement hours from batch directory files

        try:
            hours = sorted(set([int(re.search(r"(\d+)hr", filepath).group(1)) for filepath in glob.glob(f'{self.batch_dir}*')]))
            logger.info(f"Measurement hours: {hours}")
            return hours
        except Exception as e:
            logger.error(f"Failed to get measurement hours: {e}")
            return []

    def gen_df(self):

        data = pd.DataFrame(columns=['substrate', 'pixel', 'config', 'time (hrs)', 'Jsc (mA cm-2)', 'Voc (V)', 'PCE (%)', 'FF (%)'])        
        
        # list of substrates without heading 'Substrate' or nans and their pixels
        substrate_pixel = [x for x in product(list(self.substrates.iloc[0].dropna())[1:], [1,2,3,4,5,6])]
        data[['substrate', 'pixel']] = substrate_pixel
        
        # add config
        for substrate, pixel in substrate_pixel: # zip(data['substrate'], data['pixel']):
            channel = self.substrates.columns[(self.substrates == substrate).any()].tolist()[0]
            data.loc[(data['substrate']==substrate) & (data['pixel']==pixel), 'config'] = list(self.config.loc[self.config['Pixel'] == pixel, channel])[0]

        # add hours
        n = len(data)
        data = data.loc[data.index.repeat(len(self.hrs))].reset_index(drop=True) # duplicate each row n times, where n = len(hrs)
        data['time (hrs)'] = self.hrs*n

        return data

    def pop_df(self, n=5, Jsc_factor = 4000, Voc_factor = 1, PCE_factor = 4000, FF_factor = 100, save=True, save_path=None):
      
        logger.info('Calculating Jsc, Voc, PCE...')
        for substrate, pixel, h in zip(self.data['substrate'], self.data['pixel'], self.data['time (hrs)']):

            try:
                file_str = f'{self.batch_dir}_{h}hr*/*{substrate}_device{pixel}'
                it_file = glob.glob("".join([file_str, '*.it.tsv']))[0]
                vt_file = glob.glob("".join([file_str, '*.vt.tsv']))[0]
                mppt_file = glob.glob("".join([file_str, '*.mppt.tsv']))[0]

                Jsc_df = pd.read_csv(it_file, sep='\t')[-n:].median()
                Voc_df = pd.read_csv(vt_file, sep='\t')[-n:].median()
                pce_df = pd.read_csv(mppt_file, sep='\t')[-n:].median()

                # calculate metrics
                Jsc = abs(Jsc_df['current (A)']*Jsc_factor)
                Voc = abs(Voc_df['voltage (V)']*Voc_factor)
                PCE = abs(pce_df['current (A)']*pce_df['voltage (V)']*PCE_factor)
                FF = (PCE / (Jsc*Voc)) * FF_factor

                self.data.loc[((self.data['substrate']==substrate) & (self.data['pixel']==pixel) & (self.data['time (hrs)']==h)), ['Jsc (mA cm-2)', 'Voc (V)', 'PCE (%)', 'FF (%)']] = [Jsc, Voc, PCE, FF]
            
            except IndexError:
                logger.error(f'No data for {substrate} - {pixel} at {h} hrs.')

        if save:
            self.data.to_csv(save_path, index=False)
            logger.info(f'Saved data to {save_path}')

    def plot_overview(self, thresholds0, thresholds_all, config_dict, save=False, save_path=None, title=None):

        logger.info('Plotting overview...')

        # remove pixels with data outside of valid ranges
        for measure, threshold in thresholds_all.items():

            # Identify valid substrate-pixel pairs where measures are within range for all t
            valid_pixels = self.data.groupby(['substrate', 'pixel'])[measure].apply(lambda x: x.between(min(threshold), max(threshold)).all())

            # Keep only valid (substrate, pixel) pairs
            valid_pixels = valid_pixels[valid_pixels].index  # Get valid substrate-pixel pairs

            # Filter the DataFrame to keep only valid rows
            self.data = self.data[self.data.set_index(['substrate', 'pixel']).index.isin(valid_pixels)]

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        handles = {}

        for config, config_color in config_dict.items():

            for i, (measure, measure_thresh) in enumerate(thresholds0.items()):

                ax = axs.flatten()[i]

                # Filter OC pixels at t=0 and get the subset of rows where 'Voc' > Voc_thresh

                df_config0 = self.data[(self.data['time (hrs)'] == 0) & (self.data['config'] == config)]  # Select OC pixels at t=0 only
                filtered_df0 = df_config0[(df_config0[measure] > min(measure_thresh)) & (df_config0[measure] < max(measure_thresh))][['substrate', 'pixel']]  # Filter by Voc > Voc_thresh

                # Create a list of (substrate, pixel) tuples that meet the criteria
                filtered_list = list(filtered_df0.itertuples(index=False, name=None))

                # Plot Voc time series for each (substrate, pixel) pair in filtered_list
                for substrate, pixel in filtered_list:
                    sub_px_df = self.data[(self.data['substrate'] == substrate) & (self.data['pixel'] == pixel)]  # Subset for each substrate-pixel pair
                    ax.plot(sub_px_df['time (hrs)'], sub_px_df[measure], color=config_color, alpha=0.1)  # Plot the data for each pair

                # Select rows where (substrate, pixel) is in filtered_list and convert numeric columns to numeric types
                numeric_cols = ['time (hrs)', 'Jsc (mA cm-2)', 'Voc (V)', 'PCE (%)', 'FF (%)']
                df_filtered = self.data[self.data[['substrate', 'pixel']].apply(tuple, axis=1).isin(filtered_list)].copy()  # Create a copy to avoid SettingWithCopyWarning
                df_filtered[numeric_cols] = df_filtered[numeric_cols].apply(pd.to_numeric)  # Convert numeric columns to appropriate data types

                # Compute mean values of numeric columns grouped by 'time (hrs)'
                df_mean = df_filtered.groupby('time (hrs)').mean(numeric_only=True).reset_index()

                # Plot the mean Voc time series across all pixels
                mean_line, = ax.plot(df_mean['time (hrs)'], df_mean[measure], color=config_color, alpha=1, linewidth=2)  # Plot the mean data
                ax.set_xlabel('time (hrs)')
                ax.set_ylabel(measure)
                ax.set_ylim(bottom=0)
                
                handles[config] = mean_line
        
        fig.legend(handles=handles.values(), labels=handles.keys(), title='Configuration', title_fontsize=10, bbox_to_anchor=(1, 0.8), fontsize=10, frameon=False)

        if title:
            fig.suptitle(title, fontsize=16)

        if save:
            plt.savefig(save_path)
            logger.info(f'Saved plot to {save_path}')
        else:
            logger.info('Plotting complete.')
        
        return fig, axs #, handles
    
    def plot_IV(self, save=False, save_path=None):

        logger.info('Plotting IV curves...')
        
        for config, config_colors in {'SC': sns.color_palette('Blues', len(self.hrs)), 
                                    'OC': sns.color_palette('Reds', len(self.hrs)),
                                    'MPP': sns.color_palette('Greens', len(self.hrs)), }.items():
            
            df_config = self.data[self.data['config'] == config] # filter by configuration
            substrates = set(df_config['substrate'])
            pixels = set(df_config['pixel'])
            
            for substrate, pixel in product(substrates, pixels):

                fig = plt.figure(figsize=(10, 5))
                for i, h in enumerate(self.hrs):
                    # substrate, pixel = name.split('-')

                    # Read data for the corresponding hour, substrate, and pixel
                    files = glob.glob(f'{self.batch_dir}_{h}hr*/*{substrate}_device{pixel}*.liv2.tsv')
                    
                    if len(files) == 0:
                        logger.error(f'No data for {substrate} - {pixel} at {h} hrs.')
                        continue
                
                    else:
                        liv_fwd = pd.read_csv(files[0], sep='\t')

                    # # Get row and column index for subplot
                    # row, col = convert_indices(j + 1, num_cols)
                    plt.plot(liv_fwd['voltage (V)'], liv_fwd['current (A)']*4000, label=f'{h}hrs', color=config_colors[i])
                    # plt.title(f'{name} {hold}')
                    plt.ylim(-2)
                    plt.legend()

                plt.xlabel('Voltage (V)')
                plt.ylabel('Current (mA)')
                plt.title(f'{self.batch_name} {config}: {substrate}-{pixel}')

                if save:
                    plt.savefig(f'{save_path}/{config}_{substrate}_{pixel}.png')
                    logger.info(f'Saved plot to {save_path}/{config}_{substrate}_{pixel}.png')
                else:
                    logger.info('Plotting complete.')
            
        return fig