import numpy as np
import pandas as pd

import pandas as pd



def initialize_matrix():
    # Since the user wants a 57x57 table, we'll create a DataFrame with that shape.
    # The first column and row should have specific values that the user has provided.
    # The rest of the cells should be filled with NaNs for now.

    # Define the values for the first column and row (excluding the first cell, which will be NaN).
    index_and_columns = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 602, 615, 617, 
        619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 633, 637, 640, 645, 646, 647, 
        649, 650, 651, 652, 653, 654, 766, 767, 2061, 2125, 2136, 2137, 2142, 2146, 2147, 
        2148, 2166, 2197
    ]

    # Initialize the DataFrame.
    df = pd.DataFrame(index=index_and_columns, columns=index_and_columns)

    # Fill the diagonal with zeros since it represents the distance from a point to itself.
    for i in index_and_columns:
        df.at[i, i] = 0

    # Display the DataFrame to ensure it's initialized correctly.
    df.head()  # Display just the first few rows for brevity.


    np.random.seed(0)  # For reproducibility of the random numbers
    for row in index_and_columns:
        for col in index_and_columns:
            if row != col:
                # Random value assignment, can be adjusted if needed
                df.at[row, col] = np.random.uniform(0, 120)

    df.to_csv("E:/1-RL/research/subtask/BY_Nobuild_vs_Build_ODME_test/BY_Nobuild_vs_Build_ODME/demand.csv")
    # print(df)


def get_error(file_path=None):

    df = pd.read_csv(file_path, na_values='', keep_default_na=False).fillna(0)

    # Extract the 'volume' and 'obs_count' columns
    volume = df['volume'].tolist()
    obs_count = df['obs_count'].tolist()


    len_vol = len(volume)
    len_obs = len(obs_count)

    if len_vol == len_obs:
        for i in range(len_vol):
            flag = 0
            if volume[i] == 0:
                flag = 1
            if obs_count[i] == 0:
                flag = 1
            if flag == 1:
                volume[i] = 0
                obs_count[i] = 0

    filtered_volume = volume
    filtered_obs_count = obs_count

    if len(filtered_volume) != len(filtered_obs_count):
        raise ValueError("Lists must have the same length")

    # Calculate the squared differences and sum them up
    squared_diff_sum = sum((x - y) ** 2 for x, y in zip(filtered_volume, filtered_obs_count))

    # Divide the sum by the number of elements to get the mean squared error
    mse = squared_diff_sum / len(filtered_volume)

    return mse

# initialize_matrix()





def extrac_column_info(file_path=None):
    """
    this func is used for extract o-d matrix which is in the column format
    o_zone_id	d_zone_id	volume
    1	2	518.19
    1	3	568.73
    1	3	459.72
    1	4	3472.91
    1	5	117.94
    1	6	993.93
    1	8	9.43
    1	9	9.43
    """
    index_and_columns = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '602', '615', '617', 
        '619', '620', '621', '622', '623', '624', '625', '626', '627', '628', '633', '637', '640', '645', '646', '647', 
        '649', '650', '651', '652', '653', '654', '766', '767', '2061', '2125', '2136', '2137', '2142', '2146', '2147', 
        '2148', '2166', '2197'
        ]

    # Initialize the DataFrame.
    df = pd.DataFrame(index=index_and_columns, columns=index_and_columns)

    df.columns = df.columns.astype(int)
    df.index = df.index.astype(int)

    exist_file = pd.read_excel(file_path, engine='openpyxl')

    for index, row in exist_file.iterrows():
        o_id = int(row['o_zone_id'])
        d_id = int(row['d_zone_id'])
        volume = row['volume']
        if d_id in df.columns and o_id in df.index:
            df.at[o_id, d_id] = volume


    df_filled = df.fillna(0)
    return df_filled 
    # print(exist_file) 


# data = extrac_column_info()
# print(data)