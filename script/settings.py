import torch
import os

device = "cpu"
data_type = torch.float32

def get_gurobi_params():
    wlsaccessID = os.getenv('GRB_WLSACCESSID', 'undefined')
    licenseID = os.getenv('GRB_LICENSEID', '0')
    wlsSecrets = os.getenv('GRB_WLSSECRET', 'undefined')
    p = {
        'WLSACCESSID': wlsaccessID,
        'LICENSEID': int(licenseID),
        'WLSSECRET': wlsSecrets,
        'CSCLIENTLOG': int(3),
        "LogToConsole": 0
    }

    return p