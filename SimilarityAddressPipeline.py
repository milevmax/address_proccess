import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Literal
from util import df_to_nn_model_input, df_process_address_chars, calc_similarity_pair_fuzzy


FUZZY_COLUMN = 'fuzzy_score'
NN_COLUMN = 'nn_score'

class AddressComparator:
    def __init__(
            self,
            file_path: str = None,
            output_path: str = None,
            nn_path: str = None,
            method: Literal['nn', 'combo'] = 'combo',  # Specify allowed values for 'method'
            save_scores: bool = True
    ):
        self.file_path = file_path
        self.output_path = output_path
        self.save_scores = save_scores
        self.address_column_names = ('address_1', 'address_2')
        self.result_column = 'prediction'
        self.df = self._upload_file()
        if nn_path:
            self.nn = tf.keras.models.load_model(nn_path)
            self.fuzzy_only = False
        else:
            self.fuzzy_only = True
        self.char_processed = False
        self.fuzzy_threshold_high = 0.85
        self.fuzzy_threshold_low = 0.25
        self.method = method

    def _check_columns(self, df):
        if self.address_column_names[0] not in df.columns or self.address_column_names[1] not in df.columns:
            raise ValueError(f"The DataFrame is missing the following required columns: {self.address_column_names}")

    def _upload_file(self, *args):
        if len(args) == 1 and isinstance(args[0], pd.DataFrame):
            df = args[0]
            self._check_columns(df)
            self.df = df
        elif isinstance(self.file_path, str):
            df = pd.read_csv(self.file_path)
            self._check_columns(df)
            return df
        else:
            print('no data uploaded!')

    def calculate_fuzzy_scores(self):
        self.df[FUZZY_COLUMN] = self.df.apply(
            lambda row: calc_similarity_pair_fuzzy(row['address_1'], row['address_2']), axis=1)

    def prepare_data(self):
        self.df = df_process_address_chars(self.df)

    def set_same_flag(self):
        if self.method == 'combo':
            self.df[self.result_column] = np.nan
            self.df[self.result_column] = np.where(self.df[FUZZY_COLUMN] > self.fuzzy_threshold_high, 1, np.nan)
            self.df[self.result_column] = np.where(
                (self.df[FUZZY_COLUMN] < self.fuzzy_threshold_low) & (self.df[self.result_column].isnull()), 0, self.df[self.result_column]
            )
            if NN_COLUMN in self.df.columns:
                self.df[self.result_column] = self.df[self.result_column].fillna((self.df[NN_COLUMN] > 0.5).astype(int))
            else:
                self.df[self.result_column] = self.df[self.result_column].fillna(0)

        elif self.method == 'nn' and NN_COLUMN in self.df.columns:
            self.df[self.result_column] = (self.df[NN_COLUMN] > 0.5).astype(int)

    def save_file(self):
        if self.output_path:
            if not self.save_scores:
                save_df = self.df[[*self.address_column_names, self.result_column]]
                save_df.to_csv(self.output_path)
            else:
                self.df.to_csv(self.output_path)
        else:
            print('output_path is not defined!')

    def run(self):
        if not self.char_processed:
            self.prepare_data()
            self.char_processed = True
        self.calculate_fuzzy_scores()

        if self.char_processed and not self.fuzzy_only:
            nn_data = df_to_nn_model_input(self.df)
            self.df[NN_COLUMN] = self.nn.predict(nn_data)

        self.set_same_flag()
        self.save_file()


## Example of usage:
if __name__ == '__main__':
    # 1) for uploading from dir and saving result to chosen directory
    ac = AddressComparator(
        file_path='file_with_addresses.csv', # define your correct path
        output_path='result_file.csv', # file to save result
        nn_path='tf_nn_similarity_model.keras',
        method='combo') # or 'nn'
    ac.run()

    # 2) for importing AddressComparator as module and using in your own project
    #   (can be usefully for debugging or modification)
    ac = AddressComparator(
        nn_path='tf_nn_similarity_model.keras',
        method='combo') # or 'nn'

    flow_df = pd.DataFrame({
        "address_1":[
            "місто Київ, оболонський р-н, оболонський проспект 19, кв 122",
            "місто Київ, оболонський р-н, оболонський проспект 19, кв 122",
            "Луганськ, квартал Лєнінського комсомолу, буд 8а кв 101",
            "село Варені яйця курячого району, вулиця жовткова, буд 88к1"
    ],
        "address_2":[
            "місто Київ, святошинський район, вул вацлава гавела 2, кв 3",
            "місто Київ, оболонський р-н, оболонський проспект 12, кв 77",
            "м. Луганськ кв. лєнінського комсомолу, б 8-а квартира 101",
            "район курячій, с. Варені яйця , в. жовткова, будинок 88к1"
        ]
    })
    ac._upload_file(flow_df)
    ac.run()
    flow_df_with_predictions = ac.df
