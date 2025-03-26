import json, os, sys, time, io
import os.path as osp
from pathlib import Path
import importlib
from typing import List, Dict, Callable, Union
import base64
import traceback

import requests
from PIL import Image
import cv2
import numpy as np
from natsort import natsorted

IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg', '.webp']

NP_INT_TYPES = (np.int_, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64)
if int(np.version.full_version.split('.')[0]) == 1:
    NP_BOOL_TYPES = (np.bool_, np.bool8)
    NP_FLOAT_TYPES = (np.float_, np.float16, np.float32, np.float64)
else:
    NP_BOOL_TYPES = (np.bool_, np.bool)
    NP_FLOAT_TYPES = (np.float16, np.float32, np.float64)

def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__, ensure_ascii=False))

def serialize_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.ScalarType):
        if isinstance(obj, NP_BOOL_TYPES):
            return bool(obj)
        elif isinstance(obj, NP_FLOAT_TYPES):
            return float(obj)
        elif isinstance(obj, NP_INT_TYPES):
            return int(obj)
    return obj

def json_dump_nested_obj(obj, **kwargs):
    def _default(obj):
        if isinstance(obj, (np.ndarray, np.ScalarType)):
            return serialize_np(obj)
        return obj.__dict__
    return json.dumps(obj, default=lambda o: _default(o), ensure_ascii=False, **kwargs)

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.ScalarType)):
            return serialize_np(obj)
        return json.JSONEncoder.default(self, obj)

def find_all_imgs(img_dir, abs_path=False, sort=False):
    imglist = []
    for filename in os.listdir(img_dir):
        file_suffix = Path(filename).suffix
        if file_suffix.lower() not in IMG_EXT:
            continue
        if abs_path:
            imglist.append(osp.join(img_dir, filename))
        else:
            imglist.append(filename)

    if sort:
        imglist = natsorted(imglist)
        
    return imglist

def find_all_files_recursive(tgt_dir: Union[List, str], ext: Union[List, set], exclude_dirs=None):
    if isinstance(tgt_dir, str):
        tgt_dir = [tgt_dir]
    
    if exclude_dirs is None:
        exclude_dirs = set()

    filelst = []
    for d in tgt_dir:
        for root, _, files in os.walk(d):
            if osp.basename(root) in exclude_dirs:
                continue
            for f in files:
                if Path(f).suffix.lower() in ext:
                    filelst.append(osp.join(root, f))
    
    return filelst

def imread(imgpath, read_type=cv2.IMREAD_COLOR):
    if not osp.exists(imgpath):
        return None
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)
    return img

def imwrite(img_path, img, ext='.png', quality=100):
    suffix = Path(img_path).suffix
    ext = ext.lower()
    assert ext in IMG_EXT
    if suffix != '':
        img_path = img_path.replace(suffix, ext)
    else:
        img_path += ext
    
    encode_param = None
    if ext in {'.jpg', '.jpeg'}:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.webp':
        encode_param = [cv2.IMWRITE_WEBP_QUALITY, quality]

    cv2.imencode(ext, img, encode_param)[1].tofile(img_path)

def show_img_by_dict(imgdicts):
    for keyname in imgdicts.keys():
        cv2.imshow(keyname, imgdicts[keyname])
    cv2.waitKey(0)

def text_is_empty(text) -> bool:
    if isinstance(text, str):
        if text.strip() == '':
            return True
    if isinstance(text, list):
        for t in text:
            t_is_empty = text_is_empty(t)
            if not t_is_empty:
                return False
        return True    
    elif text is None:
        return True
    
def empty_func(*args, **kwargs):
    return

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_module_from_str(module_str: str):
    return importlib.import_module(module_str, package=None)

def build_funcmap(module_str: str, params_names: List[str], func_prefix: str = '', func_suffix: str = '', fallback_func: Callable = None, verbose: bool = True) -> Dict:
    
    if fallback_func is None:
        fallback_func = empty_func

    module = get_module_from_str(module_str)

    funcmap = {}
    for param in params_names:
        tgt_func = f'{func_prefix}{param}{func_suffix}'
        try:
            tgt_func = getattr(module, tgt_func)
        except Exception as e:
            if verbose:
                print(f'failed to import {tgt_func} from {module_str}: {e}')
            tgt_func = fallback_func
        funcmap[param] = tgt_func

    return funcmap

def _b64encode(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")

def img2b64(img):
    """
    Convert a PIL image to a base64-encoded string.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    return _b64encode(buffered.getvalue())

def save_encoded_image(b64_image: str, output_path: str):
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))

def submit_request(url, data, exist_on_exception=True, auth=None, wait_time = 5):
    response = None
    try:
        while True:
            try:
                response = requests.post(url, data=data, auth=auth)
                response.raise_for_status()
                break
            except Exception as e:
                if wait_time > 0:
                    print(traceback.format_exc(), file=sys.stderr)
                    print(f'sleep {wait_time} sec...')
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        if response is not None:
            print('response content: ' + response.text)
        if exist_on_exception:
            exit()
    return response