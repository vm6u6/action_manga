from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer

class M2M_translate():
    def __init__(self):
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    def run(self, text, src_lang, target_lang):
        self.tokenizer.src_lang = src_lang
        encoded_en = self.tokenizer(text, return_tensors="pt")

        generated_tokens = self.model.generate(**encoded_en, forced_bos_token_id=self.tokenizer.get_lang_id(target_lang))
        res_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        print(res_text)
        return res_text
    
class gpt_translate():
    def __init__(self, my_api):
        gpt_api = my_api
    
    def run():
        return 
    
if __name__ == "__main__":
    test = "視界が利かない森の中じゃ"
    src_lang = "ja"
    target_lang = "zh"


    a = M2M_translate()
    a.run(test, src_lang, target_lang)