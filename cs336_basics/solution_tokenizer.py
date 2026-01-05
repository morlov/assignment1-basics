import regex
import pickle
from collections.abc import Iterable, Iterator


class BPETokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None,):
        
        self.vocab = vocab
        self.merges = merges

        self.special_tokens = set(special_tokens) if special_tokens else set()
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.special_pattern = None

        if self.special_tokens:
            sorted_special = sorted(special_tokens, key=len, reverse=True)
            escaped_special = [regex.escape(token) for token in sorted_special]
            self.special_pattern = rf"({'|'.join(escaped_special)})"

    def _apply_merges(self, tokenized_text):
        for k, word in enumerate(tokenized_text):
            if b''.join(word).decode() in self.special_tokens:
                tokenized_text[k] = (b''.join(word), )
            else:
                merged = list(word)
                for merge in self.merges:
                    i = 0
                    while i < len(merged) - 1:
                        if (merged[i], merged[i + 1]) == merge:
                            merged[i] = merge[0] + merge[1]
                            del merged[i + 1]
                        else:
                            i += 1
                tokenized_text[k] = tuple(merged)
        return tokenized_text


    def encode(self, text: str):
        # Pretokenize text

        tokenized_text = []

        if self.special_pattern:
            # Split text on special tokens, keeping the delimiters
            parts = regex.split(self.special_pattern, text)
        else:
            parts = [text]


        for part in parts:
            if not part: 
                continue
                
            if part in self.special_tokens:
                tokenized_text.append((part.encode('utf-8'), ))
            else:
                matches = regex.finditer(self.pattern, part, regex.UNICODE)
                tokens = [match.group().encode('utf-8') for match in matches]
                for token in tokens:
                    tokenized_text.append(tuple(bytes([b]) for b in token))
        
        # Apply merges
        merged_text = self._apply_merges(tokenized_text)

        # Encode with vocab
        inversed_vocab =  dict(zip(self.vocab.values(), self.vocab.keys()))

        encoded_text = [inversed_vocab[token] for word in merged_text for token in word]

        return encoded_text
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            encoded_text = self.encode(text)
            for token in encoded_text:
                yield token

    def decode(self, ids: list[int]):
        
        decoded_tokens = []
        for id in ids:
            if id in self.vocab:
                token = self.vocab[id]
            else:
                token = b'\xef\xbf\xbd'
            decoded_tokens.append(token)
        return b''.join(decoded_tokens).decode(errors='replace')
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
            
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
            
        return cls(vocab, merges, special_tokens)