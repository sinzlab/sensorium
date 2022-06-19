import pandas as pd


def get_df_for_scores(session_dict, measure_attribute='score'):
    data_keys, values = [], []
    for key, unit_array in session_dict.items():
        for value in unit_array:
            data_keys.append(key)
            values.append(value)
    df = pd.DataFrame({'dataset':data_keys, measure_attribute:values})
    return df
