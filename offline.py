from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
from elasticsearch import Elasticsearch
import glob
from os import listdir
from os.path import isfile, join
import argparse
import json 

 
import base64 # convert image to b64 for indexing
 
elastic_client = Elasticsearch("http://localhost:9200",
    basic_auth=("elastic", "changeme"))

# create the "your_index_name" index for Elasticsearch if necessary
 
resp = elastic_client.indices.create(
    index = "vgg-elk",
    ignore = 400 # ignore 400 already exists code
    )
print ("\nElasticsearch create() index response -->", resp)
 
if __name__ == '__main__':
    fe = FeatureExtractor()
    img_nb=1
    _index= "vgg-elk"
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
        # create a new dict obj for the Elasticsearch doc
        _source = {}
        _source["image_path"] = str(img_path)
    # put the encoded features into the _source dict
        _source["raw_data"] =feature.tolist()
        print(_source)
 
        _id = img_nb
    # call the Elasticsearch client's index() method
        resp = elastic_client.index(
               index = _index,
               id = _id,
               body = json.dumps(_source),
               request_timeout=60)

        print ("\nElasticsearch index() response -->", resp)
        img_nb += 1
 