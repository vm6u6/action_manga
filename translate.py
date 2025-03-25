import os
import time
import re
from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple
import logging
import google.generativeai as genai
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('translator')

class M2M_translate:
    def __init__(self, model_size: str = "418M"):
        logger.info(f"Init M2M100, model size: {model_size}")
        model_path = f"facebook/m2m100_{model_size}"
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_path)

    def run(self, text: Union[str, List[str]], src_lang: str, target_lang: str) -> List[str]:
        self.tokenizer.src_lang = src_lang
        
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        logger.info(f"Translate {len(texts)} lines. {src_lang} to {target_lang}")
        
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True)
        generated_tokens = self.model.generate(
            **encoded, 
            forced_bos_token_id=self.tokenizer.get_lang_id(target_lang)
        )
        
        results = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        logger.info(f"Completed. Find {len(results)} results")
        
        return results

class LLM_translate:
    def __init__(self, 
                 api_config: Optional[Dict] = None,
                 max_rpm: int = 5,
                 model_name: str = "gemini-2.0-pro",
                 provider: str = "Google"):
        self.provider = provider
        self.model_name = model_name
        self.max_rpm = max_rpm
        self.api_keys = []
        
        if api_config is None:
            api_config = "LLM.txt"
            
        if isinstance(api_config, str):
            self._load_api_keys_from_file(api_config)
        elif isinstance(api_config, list):
            self.api_keys = api_config
        else:
            raise ValueError("api_config must in txt path or in API key list")
            
        self.request_count = 0
        self.last_reset_time = time.time()
        self.current_key_index = 0
        self.key_usage = {}
        
        self._initialize_model()
        
        logger.info(f"[INFO] Provider: {provider}, Model: {model_name}, API key num: {len(self.api_keys)}")
        
    def _load_api_keys_from_file(self, file_path: str):
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"API key not exist: {file_path}")
                
            with open(path, 'r') as f:
                content = f.read().strip()
                
            keys = [k.strip() for k in re.split(r'[;\n]', content) if k.strip()]
            self.api_keys = keys
            
            logger.info(f"From {file_path} load  {len(keys)} API key")
        except Exception as e:
            logger.error(f"Loading API key field: {e}")
            raise
            
    def _initialize_model(self):
        if not self.api_keys:
            raise ValueError("Not find and API KEY")
            
        if self.provider == "Google":
            api_key = self._select_api_key()
            genai.configure(api_key=api_key)
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
            )
            logger.info(f"Init Google model: {self.model_name}")
        else:
            raise ValueError(f"Not support: {self.provider}")
            
    def _select_api_key(self) -> str:
        if not self.api_keys:
            raise ValueError("Not find and API KEY")
            
        if len(self.api_keys) == 1:
            return self.api_keys[0]
            
        for _ in range(len(self.api_keys)):
            index = self.current_key_index % len(self.api_keys)
            key = self.api_keys[index]
            
            self._respect_key_limit(key)
            
            count, start_time = self.key_usage.get(key, (0, time.time()))
            self.key_usage[key] = (count + 1, start_time)
            
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            return key
            
    def _respect_key_limit(self, key: str):
        if self.max_rpm <= 0:
            return
            
        count, start_time = self.key_usage.get(key, (0, time.time()))
        now = time.time()
        
        if now - start_time >= 60:
            self.key_usage[key] = (0, now)
            return
            
        if count >= self.max_rpm:
            wait_time = 60 - (now - start_time)
            logger.warning(f"API key {key[:6]}... reach the limited。Wait for {wait_time:.2f} sec...")
            time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            
    def _wait_for_quota(self):
        current_time = time.time()
        time_passed = current_time - self.last_reset_time
        
        if time_passed >= 120:
            self.request_count = 0
            self.last_reset_time = current_time
            return
        
        if self.request_count >= self.max_rpm:
            wait_time = 60 - time_passed
            logger.warning(f"[INFO] Wait for {wait_time:.2f} sec...")
            time.sleep(wait_time)
            
            self.request_count = 0
            self.last_reset_time = time.time()

    def _translate_text_batch(self, 
                             texts: List[str], 
                             source_lang: str, 
                             target_lang: str, 
                             max_retries: int) -> List[str]:
        logger.info(f"Batch translate {len(texts)} lines  from {source_lang} to {target_lang}")
        
        texts_with_index = [f"[{i+1}] {t}" for i, t in enumerate(texts)]
        combined_text = "\n\n".join(texts_with_index)
        
        prompt = self._generate_batch_translation_prompt(combined_text, source_lang, target_lang)
        
        retry_count = 0
        retry_delay = 1
        
        while retry_count <= max_retries:
            try:
                self._wait_for_quota()
                
                if retry_count > 0:
                    api_key = self._select_api_key()
                    genai.configure(api_key=api_key)
                
                response = self.model.generate_content(prompt)
                self.request_count += 1
                
                result_text = response.text.strip()
                
                pattern = r'\[(\d+)\](.*?)(?=\[\d+\]|$)'
                matches = re.findall(pattern, result_text, re.DOTALL)
                
                results = [""] * len(texts)
                for idx_str, translated in matches:
                    idx = int(idx_str) - 1
                    if 0 <= idx < len(results):
                        results[idx] = translated.strip()
                
                if all(results) and len(results) == len(texts):
                    logger.info(f"Completed. Get {len(results)} res")
                    return results
                
                missing = len([r for r in results if not r])
                logger.warning(f"Field. Missing {missing} res   Retry...")
                
                retry_count += 1
                retry_delay = min(retry_delay * 2, 30)
                time.sleep(retry_delay)
                
            except Exception as e:
                retry_count += 1
                retry_delay = min(retry_delay * 2, 30)
                
                logger.warning(f"Batch Field: {e}   Retry ({retry_count}/{max_retries})   Wait {retry_delay} sec...")
                time.sleep(retry_delay)
        
        logger.warning("Batch Field: switch mode...")
        return [self.translate_text(single_text, source_lang, target_lang) for single_text in texts]
    
    def _generate_single_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        return f"""
            请将以下{source_lang}文本翻译成{target_lang}。
            保持原文的意思、风格和语气。只返回翻译结果   不需要解释。

            舉例:
            source:
                - 二人のちゅーを 目撃した ぼっちちゃん
                - ふたりさん
                - 大好きなお友達には あいさつ代わりに ちゅーするんだって
                - アイス あげた
                - 喜多ちゃんとは どどど どういった ご関係なのでしようか...
                - テレビで見た！
            target:
                - 小孤独目击了两人的接吻
                - 二里酱
                - 我听说人们会把亲吻作为与喜爱的朋友打招呼的方式
                - 我给了她冰激凌
                - 喜多酱 и你是怎么样的关系啊...
                - 我在电视上看到的！
                    
            原文:
            {text}
            
            翻译:
        """
    
    def _generate_batch_translation_prompt(self, texts: str, source_lang: str, target_lang: str) -> str:
        return f"""
            请将以下{source_lang}文本列表翻译成{target_lang}。
            保持原文的意思、风格和语气。
            请按照原文的编号格式返回翻译结果   例如"[1] 翻译结果1"   "[2] 翻译结果2"。
            只返回翻译结果   不需要解释。
            
            舉例:
            source:
                - 二人のちゅーを 目撃した ぼっちちゃん
                - ふたりさん
                - 大好きなお友達には あいさつ代わりに ちゅーするんだって
                - アイス あげた
                - 喜多ちゃんとは どどど どういった ご関係なのでしようか...
                - テレビで見た！
            target:
                - 小孤独目击了两人的接吻
                - 二里酱
                - 我听说人们会把亲吻作为与喜爱的朋友打招呼的方式
                - 我给了她冰激凌
                - 喜多酱 и你是怎么样的关系啊...
                - 我在电视上看到的！


            原文:
            {texts}
            
            翻译:
        """
    



    def translate_text(self, 
                      text: Union[str, List[str]], 
                      source_lang: str = "Japanese", 
                      target_lang: str = "Chinese", 
                      max_retries: int = 5) -> Union[str, List[str]]:
        if isinstance(text, list):
            return self._translate_text_batch(text, source_lang, target_lang, max_retries)
        
        prompt = self._generate_single_translation_prompt(text, source_lang, target_lang)
        
        retry_count = 0
        retry_delay = 1
        
        while retry_count <= max_retries:
            try:
                self._wait_for_quota()
                
                if retry_count > 0:
                    api_key = self._select_api_key()
                    genai.configure(api_key=api_key)
                
                response = self.model.generate_content(prompt)
                self.request_count += 1
                
                translated_text = response.text.strip()
                logger.info(f"Completed: {text[:30]}... -> {translated_text[:30]}...")
                return translated_text
                
            except Exception as e:
                retry_count += 1
                retry_delay = min(retry_delay * 2, 30)
                
                logger.warning(f"Field: {e}. Retry ({retry_count}/{max_retries})   Wait {retry_delay} sec...")
                time.sleep(retry_delay)
                
                if retry_count > max_retries:
                    logger.error(f"Field: {text[:100]}... error: {e}")
                    return f"[ERROR] Field: {str(e)}"
        
        return "[ERROR] Field "
    
    def translate_txt_list(self, 
                          txt_list: List[str], 
                          source_lang: str = "Japanese", 
                          target_lang: str = "Chinese") -> List[str]:
        total = len(txt_list)
        logger.info(f"[INFO] Find {total} lines need to translate")
        
        texts_to_translate = []
        for item in txt_list:
            if isinstance(item, str) and (Path(item).exists() and Path(item).is_file()):
                logger.info(f"Loading: {item}")
                with open(Path(item), 'r', encoding='utf-8') as f:
                    texts_to_translate.append(f.read())
            else:
                texts_to_translate.append(item)
        
        logger.info(f"[INFO] Translate {total} lines in one time...")
        translated_list = self.translate_text(texts_to_translate, source_lang, target_lang)
        
        logger.info(f"[INFO] Completed {total} liness...")
        return translated_list
    
    def get_usage_stats(self) -> Dict:
        return {
            "request_count": self.request_count,
            "api_keys_count": len(self.api_keys),
            "key_usage": {key[:6] + "...": count for key, (count, _) in self.key_usage.items()}
        }


if __name__ == "__main__":
    test_texts = ["視界が利かない森の中じゃ", "森の中じゃ", "平和な時代の幕開けには丁度いいな。", 
                 "平和な時代の幕開けには丁度いいな。", "平和な時代の幕開けには丁度いいな。"]
    
    translator = LLM_translate(
        api_config="LLM.txt",
        max_rpm=3,
        model_name="gemini-2.0-pro-exp-02-05",
        provider="Google"
    )
    
    results = translator.translate_txt_list(test_texts)
    print("\nResult:")
    for i, (src, tgt) in enumerate(zip(test_texts, results)):
        print(f"{i+1}. {src} -> {tgt}")
    
    print("\nINFO:")
    print(translator.get_usage_stats())