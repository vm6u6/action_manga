import os
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer
import time
from pathlib import Path
import google.generativeai as genai
from collections import deque
from threading import Lock

class M2M_translate():
    def __init__(self):
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    def run(self, text, src_lang, target_lang):
        self.tokenizer.src_lang = src_lang
        print(text)
        encoded_en = self.tokenizer(text, return_tensors="pt")

        generated_tokens = self.model.generate(**encoded_en, forced_bos_token_id=self.tokenizer.get_lang_id(target_lang))
        res_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        print(res_text)
        return res_text
    
class LLM_translate():
    def __init__(self):
        LLM_path = "LLM.txt"
        with open(LLM_path, 'r') as f:
            LLM_api_key = f.read().strip()

        # os.environ["ANTHROPIC_API_KEY"] = LLM_api_key
        # import anthropic
        # self.client = anthropic.Anthropic()

        genai.configure(api_key=LLM_api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-pro-exp-02-05",
        )

        self.request_timestamps = deque()  
        self.rate_limit = 50
        self.time_window = 60
        self.lock = Lock() 
    
    def _check_rate_limit(self):
        """检查并控制请求频率"""
        with self.lock:
            current_time = time.time()

            while self.request_timestamps and current_time - self.request_timestamps[0] > self.time_window:
                self.request_timestamps.popleft()

            if len(self.request_timestamps) >= self.rate_limit:

                wait_time = self.time_window - (current_time - self.request_timestamps[0])
                if wait_time > 0:
                    print(f"[速率限制] 已达到每分钟 {self.rate_limit} 次请求限制，等待 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)
                    return self._check_rate_limit()
            self.request_timestamps.append(current_time)
            return True

    def translate_text(self, text, source_lang="日文", target_lang="中文"):
        self._check_rate_limit()
        prompt = f"""
            请将以下{source_lang}文本翻译成{target_lang}。
            保持原文的意思、风格和语气。只返回翻译结果，不需要解释。

            原文:
            {text}
                
            翻译:
        """
        
        response = self.model.generate_content(prompt)
        translated_text = response.text
        print(translated_text)
        return translated_text

    def translate_txt_list(self, txt_list, output_dir="translated", source_lang="日语", target_lang="中文"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        total = len(txt_list)
        print(f"[INFO] Find {total} docs")
        
        for i, item in enumerate(txt_list):
            if isinstance(item, str) and (Path(item).exists() and Path(item).is_file()):
                file_path = Path(item)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                output_file = output_path / f"{file_path.stem}_translated{file_path.suffix}"
            else:
                text = item
                output_file = output_path / f"text_{i+1}_translated.txt"
            
            print(f"[INFO] Translate {i+1}/{total}: {text[:50]}...")
            translated_text = self.translate_text(text, source_lang, target_lang)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            
            print(f"Completed: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            print(f"Saved: {output_file}")
            time.sleep(1)
            
        return


if __name__ == "__main__":
    # test = "平和な時代の幕開けには丁度いいな。"
    src_lang = "ja"
    target_lang = "zh"


    # a = M2M_translate()
    # a.run(test, src_lang, target_lang)

    test = ["視界が利かない森の中じゃ", "森の中じゃ"]
    a = LLM_translate()
    # a.translate_text(test)
    a.translate_txt_list(test)

