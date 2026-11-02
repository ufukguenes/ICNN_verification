import torch
import os

device = "cpu"
data_type = torch.float32

# Every value taken from a solver is widened by absolute + relative * |value| before it is used as a bound
# (the same defaults as dhov_rust's [solver] tolerance_absolute / tolerance_relative).
solver_tolerance_absolute = 1e-5
solver_tolerance_relative = 1e-5


def get_gurobi_params():
    """
    Parameters for a new Gurobi environment.

    A Gurobi WLS license is used when GRB_WLSACCESSID, GRB_WLSSECRET and GRB_LICENSEID are set. Otherwise, Gurobi
    finds a local license itself (GRB_LICENSE_FILE or ~/gurobi.lic). The old version always passed the WLS keys,
    with the value "undefined" when unset, which fails with a local license.
    """
    p = {
        'CSCLIENTLOG': int(0),
        "LogToConsole": 0,
        "OutputFlag": 0
    }
    wls_keys = ("GRB_WLSACCESSID", "GRB_WLSSECRET", "GRB_LICENSEID")
    if all(os.getenv(key) for key in wls_keys):
        p['WLSACCESSID'] = os.getenv('GRB_WLSACCESSID')
        p['WLSSECRET'] = os.getenv('GRB_WLSSECRET')
        p['LICENSEID'] = int(os.getenv('GRB_LICENSEID'))
    return p
