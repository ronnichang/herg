import pandas as pd
from tdc.single_pred import Tox


# training date set from Karim
def get_data():
    try:
        karim_data = Tox(name='hERG_Karim')
        data_split = karim_data.get_split("scaffold")
        print("Data loaded and split successfully.")
        print("Train data samples:", len(data_split['train']))
        print("Validation data samples:", len(data_split['valid']))
        print("Test data samples:", len(data_split['test']))
    except Exception as e:
        print("An error occurred:", str(e))

    return data_split

    """
    # external data from CardioToxNet
    ext_data_keys = ['pos','neg','new']
    ext_data = []
    for k in ext_data_keys:
        curr_data = pd.read_csv("./data_cardiotoxnet/external_test_set_" + k + ".csv")
        ext_data.append(curr_data.rename(columns={'smiles': 'Drug', 'ACTIVITY': 'Y'}))
        
    # aggregate all external datasets
    ext_data_keys.append('all')
    ext_data.append(pd.concat(ext_data).reset_index(drop=True))

    return data_split, {k: d for k, d in zip(ext_data_keys, ext_data) }    
    """