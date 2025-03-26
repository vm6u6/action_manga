import re
import time
import base64
import json
import cv2
import numpy as np
from typing import List, Optional

from openai import OpenAI
import numpy as np

from .base import register_OCR, OCRBase, TextBlock


@register_OCR('llm_ocr')
class LLM_OCR(OCRBase):
    lang_map = {
        'Auto Detect': None,  # OpenAI doesn't explicitly have "Auto Detect"
        'Afrikaans': 'af',
        'Albanian': 'sq',
        'Amharic': 'am',
        'Arabic': 'ar',
        'Armenian': 'hy',
        'Assamese': 'as',
        'Azerbaijani': 'az',
        'Bangla': 'bn',
        'Basque': 'eu',
        'Belarusian': 'be',
        'Bengali': 'bn',
        'Bosnian': 'bs',
        'Breton': 'br',
        'Bulgarian': 'bg',
        'Burmese': 'my',
        'Catalan': 'ca',
        'Cebuano': 'ceb',
        'Cherokee': 'chr',
        'Chinese (Simplified)': 'zh-CN',  # Updated to OpenAI format
        'Chinese (Traditional)': 'zh-TW',  # Updated to OpenAI format
        'Corsican': 'co',
        'Croatian': 'hr',
        'Czech': 'cs',
        'Danish': 'da',
        'Dutch': 'nl',
        'English': 'en',
        'Esperanto': 'eo',
        'Estonian': 'et',
        'Faroese': 'fo',
        'Filipino': 'fil',
        'Finnish': 'fi',
        'French': 'fr',
        'Frisian': 'fy',
        'Galician': 'gl',
        'Georgian': 'ka',
        'German': 'de',
        'Greek': 'el',
        'Gujarati': 'gu',
        'Haitian Creole': 'ht',
        'Hausa': 'ha',
        'Hawaiian': 'haw',
        'Hebrew': 'he',
        'Hindi': 'hi',
        'Hmong': 'hmn',
        'Hungarian': 'hu',
        'Icelandic': 'is',
        'Igbo': 'ig',
        'Indonesian': 'id',
        'Interlingua': 'ia',
        'Irish': 'ga',
        'Italian': 'it',
        'Japanese': 'ja',
        'Javanese': 'jv',
        'Kannada': 'kn',
        'Kazakh': 'kk',
        'Khmer': 'km',
        'Korean': 'ko',
        'Kurdish': 'ku',
        'Kyrgyz': 'ky',
        'Lao': 'lo',
        'Latin': 'la',
        'Latvian': 'lv',
        'Lithuanian': 'lt',
        'Luxembourgish': 'lb',
        'Macedonian': 'mk',
        'Malagasy': 'mg',
        'Malay': 'ms',
        'Malayalam': 'ml',
        'Maltese': 'mt',
        'Maori': 'mi',
        'Marathi': 'mr',
        'Mongolian': 'mn',
        'Nepali': 'ne',
        'Norwegian': 'no',
        'Occitan': 'oc',
        'Oriya': 'or',
        'Pashto': 'ps',
        'Persian': 'fa',
        'Polish': 'pl',
        'Portuguese': 'pt',
        'Punjabi': 'pa',
        'Quechua': 'qu',
        'Romanian': 'ro',
        'Russian': 'ru',
        'Samoan': 'sm',
        'Scots Gaelic': 'gd',
        'Serbian (Cyrillic)': 'sr-Cyrl',
        'Serbian (Latin)': 'sr-Latn',
        'Shona': 'sn',
        'Sindhi': 'sd',
        'Sinhala': 'si',
        'Slovak': 'sk',
        'Slovenian': 'sl',
        'Somali': 'so',
        'Spanish': 'es',
        'Sundanese': 'su',
        'Swahili': 'sw',
        'Swedish': 'sv',
        'Tagalog': 'tl',
        'Tajik': 'tg',
        'Tamil': 'ta',
        'Tatar': 'tt',
        'Telugu': 'te',
        'Thai': 'th',
        'Tibetan': 'bo',
        'Tigrinya': 'ti',
        'Tongan': 'to',
        'Turkish': 'tr',
        'Ukrainian': 'uk',
        'Urdu': 'ur',
        'Uyghur': 'ug',
        'Uzbek': 'uz',
        'Vietnamese': 'vi',
        'Welsh': 'cy',
        'Xhosa': 'xh',
        'Yiddish': 'yi',
        'Yoruba': 'yo',
        'Zulu': 'zu',
    }

    popular_models = [
        "OAI: gpt-4-vision-preview",
        "OAI: gpt-4",
        "OAI: gpt-3.5-turbo",
        "GGL: gemini-1.5-pro-latest",
        "GGL: gemini-2.0-flash-exp",
        "GGL: gemini-2.0-flash"
    ]

    params = {
        'provider': {
            'type': 'selector',
            'options': ['OpenAI', 'Google'],
            'value': 'OpenAI',
            'description': 'Select the LLM provider.'
        },
        'api_key': {
            'value': '',
            'description': 'Your API key.'
        },
        'endpoint': {
            'value': '',  # Default to empty, allowing provider to dictate
            'description': 'Base URL for the API. Leave empty to use provider default.'
        },
        'model': {
            'type': 'selector',
            'options': popular_models,
            'value': '',  # Default to empty, allowing provider to dictate
            'description': 'Select the model to use. Leave empty to use provider default. (Provider prefix indicates the provider).'
        },
        'override_model': {
            'value': '',
            'description': 'Specify a custom model name to override the selected model.'
        },
        'language': {
            'type': 'selector',
            'options': list(lang_map.keys()),
            'value': 'Auto Detect',
            'description': 'Language for OCR.'
        },
        'prompt': {
            'value': 'Recognize the text in this image.',
            'description': 'Default prompt for OCR.'
        },
        'system_prompt': {
            'type': 'editor',
            'value': '',
            'description': 'Optional system prompt to guide the model\'s behavior.'
        },
        'proxy': {
            'value': '',
            'description': 'Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)'
        },
        'delay': {
            'value': 0.0,
            'description': 'Delay in seconds between requests.'
        },
        'requests_per_minute': {
            'value': 0,
            'description': 'Maximum number of requests per minute (0 for no limit).'
        },
        'description': 'OCR using various LLMs compatible with the OpenAI API.'
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.last_request_time = 0
        self.client = None
        self._initialize_client()
        self.request_count_minute = 0
        self.minute_start_time = time.time()

    def _initialize_client(self):
        import httpx

        # Configure proxies using mounts
        if self.proxy:
            proxy_mounts = {
                "http://": httpx.HTTPTransport(proxy=self.proxy),
                "https://": httpx.HTTPTransport(proxy=self.proxy),
            }
            transport = httpx.Client(mounts=proxy_mounts)
        else:
            transport = httpx.Client()  # No proxy

        # Determine the endpoint
        endpoint = self.endpoint
        if not endpoint:  # If endpoint is empty, use provider default
            provider = self.provider
            if provider == 'OpenAI':
                endpoint = 'https://api.openai.com/v1'
            elif provider == 'Google':
                endpoint = 'https://generativelanguage.googleapis.com/v1beta/openai'
            else:
                endpoint = 'https://api.openai.com/v1'  # Default

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=endpoint,
            http_client=transport
        )

    @property
    def provider(self):
        return self.get_param_value('provider')

    @property
    def request_delay(self):
        try:
            return float(self.get_param_value('delay'))
        except (ValueError, TypeError):
            return 1.0

    @property
    def api_key(self):
        return self.get_param_value('api_key')

    @property
    def endpoint(self):
        return self.get_param_value('endpoint')

    @property
    def model(self):
        return self.get_param_value('model')

    @property
    def override_model(self):
        return self.get_param_value('override_model')

    @property
    def language(self):
        lang_name = self.get_param_value('language')
        return self.lang_map.get(lang_name)

    @property
    def prompt(self):
        return self.get_param_value('prompt')

    @property
    def system_prompt(self):
        return self.get_param_value('system_prompt')

    @property
    def proxy(self):
        return self.get_param_value('proxy')

    @property
    def requests_per_minute(self):
        return int(self.get_param_value('requests_per_minute'))

    def _respect_delay(self):
        current_time = time.time()

        # Handle RPM limit
        if self.requests_per_minute > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time

            if self.request_count_minute >= self.requests_per_minute:
                wait_time = 62 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    if self.debug_mode:
                        self.logger.info(f'Reached request limit. Waiting {wait_time:.2f} seconds.')
                    time.sleep(wait_time)
                # Reset the counter and start time after waiting, just in case.
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        # Handle delay parameter
        time_since_last_request = current_time - self.last_request_time
        if self.debug_mode:
            self.logger.info(f'Time since last request: {time_since_last_request} seconds')

        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            if self.debug_mode:
                self.logger.info(f'Waiting {sleep_time} seconds before next request')
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        if self.requests_per_minute > 0:
            self.request_count_minute += 1

    def ocr(self, img_base64: str, prompt_override: str = None) -> str:
        """
        Performs OCR on a base64 encoded image.
        """
        if self.debug_mode:
            self.logger.debug(f'Starting OCR on image')
        self._respect_delay()

        try:
            prompt_text = prompt_override if prompt_override else self.prompt
            if self.language:
                prompt_text += f" The language is {self.language}."

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ],
                }
            )

            # Determine the model
            model_name = self.override_model
            if not model_name:  # If override_model is empty
                model_name = self.model
                if not model_name:  # If model is also empty, determine from provider
                    provider = self.provider
                    # You might want to set default models for each provider here
                    if provider == 'OpenAI':
                        model_name = "gpt-4-vision-preview"
                    elif provider == 'Google':
                        model_name = "gemini-1.5-pro-latest"
                    else:
                        model_name = "gpt-4-vision-preview"  # Default

                # Extract model name without provider prefix if it exists
                if ': ' in model_name:
                    model_name = model_name.split(': ', 1)[1]

            # Log the model being used
            if self.debug_mode:
                self.logger.info(f"Using model: {model_name}")

            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=300,  # Adjust as needed
            )

            if response.choices:
                full_text = response.choices[0].message.content
                if self.debug_mode:
                    self.logger.debug(f'OCR result: {full_text}')
                return full_text
            else:
                if self.debug_mode:
                    self.logger.warning('No text found in OCR response')
                return ''

        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return ''

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        """
        Processes a list of text blocks in an image.
        """
        im_h, im_w = img.shape[:2]
        if self.debug_mode:
            self.logger.debug(f'Image dimensions: {im_h}x{im_w}')
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if self.debug_mode:
                self.logger.debug(f'Processing block: ({x1}, {y1}, {x2}, {y2})')
            if y2 <= im_h and x2 <= im_w and x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2:
                cropped_img = img[y1:y2, x1:x2]

                # Encode the cropped image to base64
                _, buffer = cv2.imencode('.jpg', cropped_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                if self.debug_mode:
                    self.logger.debug(f'Cropped image dimensions: {cropped_img.shape}')
                blk.text = self.ocr(img_base64, prompt_override=kwargs.get('prompt', ""))
            else:
                if self.debug_mode:
                    self.logger.warning('Invalid text block coordinates')
                blk.text = ''

    def ocr_img(self, img: np.ndarray, prompt: str = "") -> str:
        """
        Performs OCR on the entire image.
        """
        # Encode the entire image to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return self.ocr(img_base64, prompt_override=prompt)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ['api_key', 'endpoint', 'proxy', 'provider', 'model', 'override_model']:
            self._initialize_client()
