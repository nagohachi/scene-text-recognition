from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self


class TokenizerBase(ABC):
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, vocab: list[str]) -> None:
        self._vocab = self._build_vocab(vocab)
        self._token_to_id = {token: i for i, token in enumerate(self._vocab)}

    @abstractmethod
    def _get_special_tokens(self) -> list[str]:
        """Return special tokens to prepend to vocab."""
        ...

    def _build_vocab(self, vocab: list[str]) -> list[str]:
        special_tokens = self._get_special_tokens()
        return special_tokens + [t for t in vocab if t not in special_tokens]

    @property
    def pad_id(self) -> int:
        return self._token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, text: str) -> list[int]:
        return [self._token_to_id.get(c, self.unk_id) for c in text]

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for token in self._vocab:
                f.write(token + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            vocab = [line.rstrip("\n") for line in f]
        tokenizer = cls.__new__(cls)
        tokenizer._vocab = vocab
        tokenizer._token_to_id = {token: i for i, token in enumerate(vocab)}
        return tokenizer

    @classmethod
    def from_texts(cls, texts: list[str]) -> Self:
        chars = set()
        for text in texts:
            chars.update(text)
        vocab = sorted(chars)
        return cls(vocab)


class CTCTokenizer(TokenizerBase):
    BLANK_TOKEN = "<blank>"

    def _get_special_tokens(self) -> list[str]:
        return [self.BLANK_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN]

    @property
    def blank_id(self) -> int:
        return self._token_to_id[self.BLANK_TOKEN]

    def decode(self, ids: list[int]) -> str:
        tokens = []
        prev_id = None
        for id in ids:
            if id == self.pad_id:
                continue
            if id == self.blank_id:
                prev_id = id
                continue
            # CTC collapse: skip repeated tokens
            if id == prev_id:
                continue
            tokens.append(self._vocab[id])
            prev_id = id
        return "".join(tokens)


class AEDTokenizer(TokenizerBase):
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    def _get_special_tokens(self) -> list[str]:
        return [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]

    @property
    def sos_id(self) -> int:
        return self._token_to_id[self.SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[self.EOS_TOKEN]

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for id in ids:
            if id == self.pad_id:
                continue
            if id == self.sos_id:
                continue
            if id == self.eos_id:
                break
            tokens.append(self._vocab[id])
        return "".join(tokens)
