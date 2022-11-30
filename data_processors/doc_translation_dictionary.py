from fairseq.data.dictionary import Dictionary

SEP = "[@@SEP@@]"


def update_dict(d, k, v):
    s = d.get(k, set())
    assert v not in s
    s.add(v)
    d[k] = s


class DocTranslationDictionary(Dictionary):
    def __init__(self, *args, **kwargs):
        update_dict(kwargs, "extra_special_symbols", SEP)
        super().__init__(*args, **kwargs)
        self.sep_index = self.indices[SEP]

    def sep(self):
        return self.sep_index

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
    ):
        # We can't use the **kwargs trick like in __init__ because of Dictionary.string() calls this
        # method again with extra_symbols_to_ignore in *args
        if extra_symbols_to_ignore is None:
            extra_symbols_to_ignore = set()
        extra_symbols_to_ignore.add(self.sep())
        return super().string(
            tensor,
            bpe_symbol=bpe_symbol,
            escape_unk=escape_unk,
            extra_symbols_to_ignore=extra_symbols_to_ignore,
            unk_string=unk_string,
        )
