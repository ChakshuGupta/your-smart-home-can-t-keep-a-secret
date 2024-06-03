import os


def verify_config(config):
    """
    Verify the entries in the config file pass the basic sanity checks
    """
    error = False

    if "dataset-path" not in config:
        error = True
    else:
        if not os.path.exists(config["dataset-path"]):
            error = True
        elif not os.path.isdir(config["dataset-path"]):
            error = True
    
    if "device-file" not in config:
        error = True
    else:
        if not os.path.exists(config["device-file"]):
            error = True
        elif not os.path.isfile(config["device-file"]):
            error = True
        elif not config["device-file"].endswith(".txt"):
            error = True
    
    return error
