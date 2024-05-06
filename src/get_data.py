import pandas as pd
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
from tdc.single_pred.single_pred_dataset import DataLoader


# training date set from Karim
def get_data():
    seed = 42
    karim_data = Tox(name='hERG_Karim')
    karim_data = karim_data.get_split(method='random', seed=seed, frac=[1, 0, 0])
    
    wang_data = Tox(name = 'hERG')
    wang_data = wang_data.get_split(method='random', seed=seed, frac=[1, 0, 0])
    
    du_data = Tox(name = 'herg_central', label_name = 'hERG_inhib')
    du_data = du_data.get_split(method='scaffold', seed=seed, frac=[0.9, 0.05, 0.05])
    
    return {
        'karim': karim_data,
        'wang': wang_data,
        'du': du_data,
    }


def get_cardio_tox_net(): # TO DO
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


if __name__ == "__main__":
    data = get_data()
