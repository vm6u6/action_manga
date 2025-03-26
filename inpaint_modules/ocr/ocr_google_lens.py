import re
import numpy as np
import time
import cv2
import random
import string
from typing import List

import httpx
from PIL import Image
import io
import json5
import lxml.html
import http.cookiejar as cookielib

from .base import register_OCR, OCRBase, TextBlock

class LensCore:
    LENS_ENDPOINT = 'https://lens.google.com/v3/upload'
    SUPPORTED_MIMES = [
        'image/x-icon', 'image/bmp', 'image/jpeg',
        'image/png', 'image/tiff', 'image/webp', 'image/heic'
    ]
    # https://github.com/AuroraWright/owocr/blob/master/owocr/ocr.py
    HEADERS = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'max-age=0',
        'Origin': 'https://lens.google.com',
        'Referer': 'https://lens.google.com/',
        'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="131", "Chromium";v="131"',
        'Sec-Ch-Ua-Arch': '"x86"',
        'Sec-Ch-Ua-Bitness': '"64"',
        'Sec-Ch-Ua-Full-Version': '"131.0.6778.205"',
        'Sec-Ch-Ua-Full-Version-List': '"Not A(Brand";v="99.0.0.0", "Google Chrome";v="131", "Chromium";v="131"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Model': '""',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Ch-Ua-Platform-Version': '"15.0.0"',
        'Sec-Ch-Ua-Wow64': '?0',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'X-Client-Data': 'CIW2yQEIorbJAQipncoBCIH+ygEIkqHLAQiKo8sBCPWYzQEIhaDNAQji0M4BCLPTzgEI19TOAQjy1c4BCJLYzgEIwNjOAQjM2M4BGM7VzgE='
    }

    def __init__(self, proxy=None):
        self.proxy = proxy
        self.cookie_jar = cookielib.CookieJar()

    def _send_request(self, url, headers, files, params=None):
        try:
            client_kwargs = {}
            if self.proxy:
                if isinstance(self.proxy, str):
                    client_kwargs['proxy'] = self.proxy
                elif isinstance(self.proxy, dict):
                    mounts = {}
                    if 'http://' in self.proxy:
                        mounts["http://"] = httpx.HTTPTransport(proxy=self.proxy['http://'])
                    if 'https://' in self.proxy:
                        mounts["https://"] = httpx.HTTPTransport(proxy=self.proxy['https://'])
                    if mounts:
                        client_kwargs['mounts'] = mounts
                else:
                    raise ValueError("Proxy must be a string or a dictionary")
            client = httpx.Client(**client_kwargs)
            response = client.post(url, headers=headers, files=files, params=params)
            if response.status_code == 303:
                raise Exception("Error 303: See Other. Potential misconfiguration in headers or file upload.")
            if response.status_code != 200:
                raise Exception(f"Failed to upload image. Status code: {response.status_code}")
            return response
        except Exception as e:
            raise

    def scan_by_data(self, data, mime, dimensions):
        headers = self.HEADERS.copy()
        random_filename = ''.join(random.choices(string.ascii_letters, k=8)) + '.jpg'
        files = {
            'encoded_image': (random_filename, data, mime),
            'original_width': (None, str(dimensions[0])),
            'original_height': (None, str(dimensions[1])),
            'processed_image_dimensions': (None, f"{dimensions[0]},{dimensions[1]}")
        }
        params = {'ep': 'ccm', 're': 'dcsp', 's': '4', 'st': str(time.time() * 1000), 'sideimagesearch': '1', 'vpw': str(dimensions[0]), 'vph': str(dimensions[1])}
        response = self._send_request(self.LENS_ENDPOINT, headers, files, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to upload image. Status code: {response.status_code}")

        tree = lxml.html.parse(io.StringIO(response.text))
        r = tree.xpath("//script[@class='ds:1']")
        return json5.loads(r[0].text[len("AF_initDataCallback("):-2])

class Lens(LensCore):
    def __init__(self, proxy=None):
        super().__init__(proxy=proxy)

    @staticmethod
    def resize_image(image, max_size=(1000, 1000)):
        image.thumbnail(max_size)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue(), image.size

    def scan_by_file(self, file_path):
        with Image.open(file_path) as img:
            img_data, dimensions = self.resize_image(img)
        return self.scan_by_data(img_data, 'image/jpeg', dimensions)

    def scan_by_buffer(self, buffer):
        img = Image.open(io.BytesIO(buffer))
        img_data, dimensions = self.resize_image(img)
        return self.scan_by_data(img_data, 'image/jpeg', dimensions)

class LensAPI:
    def __init__(self, proxy=None):
        self.lens = Lens(proxy=proxy)

    @staticmethod
    def extract_text_and_coordinates(data):
        text_with_coords = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    for sub_item in item: # Corrected loop variable name
                        if isinstance(sub_item, list) and len(sub_item) > 1 and isinstance(sub_item[0], str):
                            word = sub_item[0]
                            coords = sub_item[1]
                            if isinstance(coords, list) and all(isinstance(coord, (int, float)) for coord in coords):
                                text_with_coords.append({"text": word, "coordinates": coords})
                        else:
                            text_with_coords.extend(LensAPI.extract_text_and_coordinates(sub_item))
                else:
                    text_with_coords.extend(LensAPI.extract_text_and_coordinates(item))
        elif isinstance(data, dict):
            for value in data.values():
                text_with_coords.extend(LensAPI.extract_text_and_coordinates(value))
        return text_with_coords

    @staticmethod
    def stitch_text_smart(text_with_coords):
        transformed_coords = [{'text': item['text'], 'coordinates': [item['coordinates'][1], item['coordinates'][0]]} for item in text_with_coords]
        sorted_elements = sorted(transformed_coords, key=lambda x: (round(x['coordinates'][1], 2), x['coordinates'][0]))

        stitched_text = []
        current_y = None
        current_line = []
        for element in sorted_elements:
            if current_y is None or abs(element['coordinates'][1] - current_y) > 0.05:
                if current_line:
                    stitched_text.append(" ".join(current_line))
                    current_line = []
                current_y = element['coordinates'][1]
            if element['text'] in [',', '.', '!', '?', ';', ':'] and current_line:
                current_line[-1] += element['text']
            else:
                current_line.append(element['text'])
        if current_line:
            stitched_text.append(" ".join(current_line))
        return "\n".join(stitched_text).strip()

    @staticmethod
    def stitch_text_sequential(text_with_coords):
        stitched_text = " ".join([element['text'] for element in text_with_coords])
        stitched_text = re.sub(r'\s+([,?.!])', r'\1', stitched_text)
        return stitched_text.strip()

    @staticmethod
    def extract_full_text(data):
        try:
            text_data = data[3][4][0][0]
            if isinstance(text_data, list):
                return "\n".join(text_data)
            return text_data
        except (IndexError, TypeError):
            return "Full text not found (or Lens could not recognize it)"

    @staticmethod
    def extract_language(data):
        try:
            return data[3][3]
        except (IndexError, TypeError):
            return "Language not found in expected structure"

    def process_image(self, image_path=None, image_buffer=None, response_method="Full Text"):
        if image_path:
            result = self.lens.scan_by_file(image_path)
        elif image_buffer:
            result = self.lens.scan_by_buffer(image_buffer)
        else:
            raise ValueError("Either image_path or image_buffer must be provided")

        text_with_coords = self.extract_text_and_coordinates(result['data'])

        if response_method == "Full Text":
            return {
                'full_text': self.extract_full_text(result['data']),
                'language': self.extract_language(result['data']),
                'text_with_coordinates': text_with_coords
            }
        elif response_method == "Coordinate sequence":
            return {
                'full_text': self.stitch_text_sequential(text_with_coords),
                'language': self.extract_language(result['data']),
                'text_with_coordinates': text_with_coords
            }
        elif response_method == "Location coordinates":
            return {
                'full_text': self.stitch_text_smart(text_with_coords),
                'language': self.extract_language(result['data']),
                'text_with_coordinates': text_with_coords
            }
        else:
            raise ValueError("Invalid response method")

def format_ocr_result(result):
    formatted_result = {
        "language": result.get("language", ""),
        "text_with_coordinates": [
            f"{item['text']}: {item['coordinates']}"
            for item in result.get("text_with_coordinates", [])
        ]
    }
    return json5.dumps(formatted_result, indent=4, ensure_ascii=False)

@register_OCR('google_lens')
class OCRLensAPI(OCRBase):
    params = {
        "delay": 1.0,
        'newline_handling': {
            'type': 'selector',
            'options': [
                'preserve',
                'remove'
            ],
            'value': 'preserve',
            'description': 'Choose how to handle newline characters in OCR result'
        },
        'no_uppercase': {
            'type': 'checkbox',
            'value': False,
            'description': 'Convert text to lowercase except the first letter of each sentence'
        },
        'response_method': {
            'type': 'selector',
            'options': [
                'Full Text',
                'Coordinate sequence',
                'Location coordinates'
            ],
            'value': 'Full Text',
            'description': 'Choose the method for extracting text from image'
        },
        'proxy': {
            'value': '',
            'description': 'Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)'
        },
        'description': 'OCR using Google Lens API'
    }

    @property
    def request_delay(self):
        try:
            return float(self.get_param_value('delay'))
        except (ValueError, TypeError):
            return 1.0

    @property
    def newline_handling(self):
        return self.get_param_value('newline_handling')

    @property
    def no_uppercase(self):
        return self.get_param_value('no_uppercase')

    @property
    def response_method(self):
        return self.get_param_value('response_method')

    @property
    def proxy(self):
        return self.get_param_value('proxy')

    def __init__(self, **params) -> None:
        if 'delay' in params:
            try:
                params['delay'] = float(params['delay'])
            except (ValueError, TypeError):
                params['delay'] = 1.0  
        super().__init__(**params)
        self.api = LensAPI(proxy=self.proxy)
        self.last_request_time = 0

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        if self.debug_mode:
            self.logger.debug(f'Image size: {im_h}x{im_w}')
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if self.debug_mode:
                self.logger.debug(f'Processing block: ({x1, y1, x2, y2})')
            if y2 < im_h and x2 < im_w and x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2:
                cropped_img = img[y1:y2, x1:x2]
                if self.debug_mode:
                    self.logger.debug(f'Cropped image size: {cropped_img.shape}')
                blk.text = self.ocr(cropped_img)
            else:
                if self.debug_mode:
                    self.logger.warning('Invalid text bbox to target image')
                blk.text = ['']

    def ocr_img(self, img: np.ndarray) -> str:
        if self.debug_mode:
            self.logger.debug(f'ocr_img: {img.shape}')
        return self.ocr(img)

    def ocr(self, img: np.ndarray) -> str:
        if self.debug_mode:
            self.logger.debug(f'Starting OCR on image of shape: {img.shape}')
        self._respect_delay()
        try:
            if img.size > 0:  # Check if the image is not empty
                if self.debug_mode:
                    self.logger.debug(f'Input image size: {img.shape}')
                _, buffer = cv2.imencode('.jpg', img)
                result = self.api.process_image(image_buffer=buffer.tobytes(), response_method=self.response_method)
                if self.debug_mode:
                    formatted_result = format_ocr_result(result)
                    self.logger.debug(f'OCR result: {formatted_result}')
                ignore_texts = [
                    'Full text not found in expected structure',
                    'Full text not found (or Lens could not recognize it)'
                ]
                if result['full_text'] in ignore_texts:
                    return ''
                full_text = result['full_text']
                if self.newline_handling == 'remove':
                    full_text = full_text.replace('\n', ' ')

                full_text = self._apply_punctuation_and_spacing(full_text)

                if self.no_uppercase:
                    full_text = self._apply_no_uppercase(full_text)

                if isinstance(full_text, list):
                    return '\n'.join(full_text)
                else:
                    return full_text
            else:
                if self.debug_mode:
                    self.logger.warning('Empty image provided for OCR')
                return ''
        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"OCR error: {str(e)}")
            return ''

    def _apply_no_uppercase(self, text: str) -> str:
        def process_sentence(sentence):
            words = sentence.split()
            if not words:
                return ''
            processed = [words[0].capitalize()] + [word.lower() for word in words[1:]]
            return ' '.join(processed)

        sentences = re.split(r'(?<=[.!?…])\s+', text)
        processed_sentences = [process_sentence(sentence) for sentence in sentences]

        return ' '.join(processed_sentences)

    def _apply_punctuation_and_spacing(self, text: str) -> str:
        text = re.sub(r'\s+([,.!?…])', r'\1', text)
        text = re.sub(r'([,.!?…])(?!\s)(?![,.!?…])', r'\1 ', text)
        text = re.sub(r'([,.!?…])\s+([,.!?…])', r'\1\2', text)
        return text.strip()

    def _respect_delay(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if self.debug_mode:
            self.logger.info(f'Time since last request: {time_since_last_request} seconds')

        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            if self.debug_mode:
                self.logger.info(f'Sleeping for {sleep_time} seconds')
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def updateParam(self, param_key: str, param_content):
        if param_key == 'delay':
            try:
                param_content = float(param_content)
            except (ValueError, TypeError):
                param_content = 1.0 # Default value
        super().updateParam(param_key, param_content)
        if param_key == 'proxy':
            # When changing the proxy, recreate the client
            self.api.lens.proxy = self.proxy # Update the proxy
            self.api.lens.client = None # Reset the client to create a new one on the next request
