from tensorflow.keras.preprocessing.sequence import pad_sequences
from fuzzywuzzy import fuzz


MAX_LEN = 120
ukrainian_alphabet = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'"
additional_chars = ' .,-/'
numbers = '0123456789'

ALLOWED_CHARS = ukrainian_alphabet + additional_chars + numbers
char_to_index = {char: idx + 1 for idx, char in enumerate(ALLOWED_CHARS)}

def tokenize_address(address):
    return [char_to_index[char] for char in address if char in char_to_index]

def process_address_chars(address, allowed_chars):
    address = address.lower()
    address = ''.join([char for char in address if char in allowed_chars])
    address = address.strip()
    return address

def df_process_address_chars(df):
    df = df.astype({'address_1': 'str', 'address_2': 'str'})
    df['address_1'] = df['address_1'].apply(lambda x: process_address_chars(x, ALLOWED_CHARS))
    df['address_2'] = df['address_2'].apply(lambda x: process_address_chars(x, ALLOWED_CHARS))
    return df

def df_to_nn_model_input(df):
    tokenized_address_1 = df['address_1'].apply(tokenize_address)
    tokenized_address_2 = df['address_2'].apply(tokenize_address)
    address_1_padded = pad_sequences(tokenized_address_1, maxlen=MAX_LEN)
    address_2_padded = pad_sequences(tokenized_address_2, maxlen=MAX_LEN)
    return address_1_padded, address_2_padded


def calc_similarity_pair_fuzzy(address_1: str, address_2: str) -> float:
    address_1, address_2 = (process_address_chars(address_1, ALLOWED_CHARS),
                            process_address_chars(address_2, ALLOWED_CHARS))
    similarity_measure = fuzz.partial_ratio(address_1, address_2)/100
    return similarity_measure
