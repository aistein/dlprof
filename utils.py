import pandas as pd
import pathlib
import requests

def download_or_get(data_url, col_names, index_col):
    data = pathlib.Path("data")
    if not data.exists():
        data.mkdir()
    p = data / pathlib.Path(data_url.split("/")[-1])
    if not p.exists():
        r = requests.get(data_url, stream=True)
        with open(p , "wb") as loc: 
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk: # filter out keep-alive new chunks
                    loc.write(chunk)
    return pd.read_csv(p,
                      header=None,
                      names=col_names,
                      index_col=index_col)