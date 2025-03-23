import os
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer


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
        anthropic_api_key = ""
        LLM_path = "LLM.txt"
        f = open(LLM_path, 'r')
        LLM_api_key = f.read()
        os.environ["ANTHROPIC_API_KEY"] = LLM_api_key
        import anthropic
        client = anthropic.Anthropic()

    def translate_text(text, source_lang="英语", target_lang="中文"):
        slef.prompt = f"""
            请将以下{source_lang}文本翻译成{target_lang}。保持原文的意思、风格和语气。只返回翻译结果，不需要解释。

            原文:
            {text}
                
            翻译:
        """
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219", 
            max_tokens=4000,  
            temperature=0,  # 
            system="您是一位专业翻译，精通多种语言之间的转换。请提供准确、自然的翻译。",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        translated_text = message.content[0].text
        return translated_text

    def translate_txt_list(txt_list, output_dir="translated", source_lang="英语", target_lang="中文"):


        return


if __name__ == "__main__":
    test = "視界が利かない森の中じゃ"
    src_lang = "ja"
    target_lang = "zh"


    # a = M2M_translate()
    # a.run(test, src_lang, target_lang)
    test = ["視界が利かない森の中じゃ", "森の中じゃ"]
    test = LLM_translate()

