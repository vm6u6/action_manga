from .base import *

import os
from typing import Literal
from msl.loadlib import Client64


class MyClient(Client64):
    def __init__(self, engine_path, engine_type: Literal['J2K', 'K2J'], dat_path):
        super(MyClient, self).__init__(module32=str(os.path.dirname(os.path.realpath(__file__))) + '\module_eztrans32.py',
                                       engine_path=engine_path,
                                       engine_type=engine_type,
                                       dat_path=dat_path)

    def translate(self, src_text: Union[str, list]):
        return self.request32('translate', src_text)


@register_translator('ezTrans')
class ezTransTranslator(BaseTranslator):
    concate_text = True

    params: Dict = {
        'path_dat': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\Dat",
        'path_j2k': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\J2KEngine.dll",
        'path_k2j(Optional)': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\ehnd-kor.dll"
    }

    def _setup_translator(self):
        self.textblk_break = '\n'
        self.lang_map['日本語'] = 'j'
        self.lang_map['한국어'] = 'k'

        self.j2k_engine = MyClient(self.params['path_j2k'], "J2K", self.params['path_dat'])
        if os.path.exists(self.params['path_k2j(Optional)']):
            self.k2j_engine = MyClient(self.params['path_k2j(Optional)'], "K2J", self.params['path_dat'])

    def _translate(self, src_list: List[str]) -> List[str]:
        source = self.lang_map[self.lang_source]
        target = self.lang_map[self.lang_target]

        if source != target:
            engine: MyClient = getattr(self, f"{source}2{target}_engine")
            return engine.translate(src_list)
        else:
            return src_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

    @property
    def supported_tgt_list(self) -> List[str]:
        return ['한국어', '日本語'] if hasattr(self,"k2j_engine") else ['한국어']

    @property
    def supported_src_list(self) -> List[str]:
        return ['한국어', '日本語'] if hasattr(self,"k2j_engine") else ['日本語']
