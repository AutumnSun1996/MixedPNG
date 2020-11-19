from urllib3.util import connection

_orig_create_connection = connection.create_connection

def patched_create_connection(address, *args, **kwargs):
    """Wrap urllib3's create_connection to resolve the name elsewhere"""
    # resolve hostname to an ip address; use your own
    # resolver here, as otherwise the system resolver will be used.
    host, port = address
    if host == "pixiv.net":
        host = "210.140.131.182"
        print("Updated")
    return _orig_create_connection((host, port), *args, **kwargs)


connection.create_connection = patched_create_connection

import requests

c = requests.session()
resp = c.get("https://pixiv.net")
print(resp)
