import pandas as pd
import pathlib
import requests
import tarfile

def download_or_get(data_url, col_names, index_col):
    data = pathlib.Path("data")
    if not data.exists():
        data.mkdir()
    p = data / pathlib.Path(data_url.split("/")[-1])
    if not p.exists():
        download(data_url, p)
    return pd.read_csv(p,
                      header=None,
                      names=col_names,
                      index_col=index_col)

def download(data_url, name):
    r = requests.get(data_url, stream=True)
    with open(name , "wb") as loc: 
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                loc.write(chunk)

def extract(fname):
    p = pathlib.Path(fname)
    with tarfile.open(fname) as tar:
        tar.extractall(p.parent)
        
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_cifar10_batches():
    dicts = []
    root_batch_name = "data/cifar-10-batches-py/data_batch_"
    for i in range(1, 6):
        batch_name = root_batch_name + str(i)
        dicts.append(cifar_10_dict_to_batch(unpickle(batch_name), (i - 1) * 10000))
    result = {}
    for d in dicts:
        result.update(d)
    return result

def cifar_10_dict_to_batch(unpickled_cifar10_batch, batch_num):
    labels = unpickled_cifar10_batch[b"labels"]
    data = unpickled_cifar10_batch[b"data"]
    return {idx + batch_num: (label, data) for idx, (label, data) in enumerate(zip(labels, data))}
