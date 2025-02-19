from logger_setup import logger
import re
import glob
import pandas as pd
import numpy as np
from itertools import product

class Batch:
    """
    Class representing a batch of devices.
    """

    def __init__(self,                     # initialise Batch class
                 batch_dir: str,           # relevant directories start with this string
                 substrates_file: str,     # path to substratesfile (excel)
                 config_file: str          # path to config file (excel)    
                 ):

        # initialise batch directory
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

    def pop_df(self, n=5, Jsc_factor = 4000, Voc_factor = 1, PCE_factor = 4000, FF_factor = 100):
      
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