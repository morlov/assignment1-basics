import os
from collections import defaultdict, Counter
import multiprocessing as mp
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(task):
    """
    Docstring for process_chunk
    
    :param task: file_path, start, end, special_tokens
    """
    file_path, start, end, special_tokens = task
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    
    regex_pattern = '|'.join(re.escape(sep) for sep in special_tokens)
    partial_counter = Counter()
    for chunk_part in re.split(regex_pattern, chunk):
        matches = re.finditer(PAT, chunk_part)
        tokens = [match.group(0).encode('utf-8') for match in matches]
        for token in tokens:
            partial_counter.update([tuple(bytes([b]) for b in token)])
    return partial_counter

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    """

    vocab = {bytes([i]): i for i in range(256)}
    merges = []

    # Add special tokens
    for special_token in special_tokens:
        idx = len(vocab)
        vocab[special_token.encode('utf-8')] = idx

    # Pretokenize
    counter = Counter()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, mp.cpu_count(), b"<|endoftext|>")

    num_processes = min(mp.cpu_count(), len(boundaries))

    # parallel implementation
    with mp.Pool(processes=num_processes) as pool:
        tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        partial_counts = pool.map(process_chunk, tasks)

    for partial_count in partial_counts:
        counter.update(partial_count)

    freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for word in counter:
        for i in range(len(word)-1):
            freqs[(word[i], word[i+1])] += counter[word]

    # Apply merges
    num_merges = vocab_size - len(vocab)

    
    for _ in range(num_merges):
        
        # Find most frequent pair
        pair = max(freqs, key=lambda k: (freqs[k], k)) # Max by count break ties lexicographically
        idx1, idx2 = pair

        idx = len(vocab)
        merges.append((idx1, idx2))

        vocab[idx1 + idx2] = idx

        # Merge pair in counter
        for word in list(counter.keys()):
            new_word = list(word)
            i = 0
            while i < len(new_word) - 1:
                if (new_word[i], new_word[i + 1]) == pair:
                    merged_word = pair[0] + pair[1]
                    if i!=0:
                        freqs[(new_word[i - 1], new_word[i])] -= counter[word]
                        freqs[(new_word[i - 1], merged_word)] += counter[word]

                    if (i+2)!=len(new_word):
                        freqs[(new_word[i+1], new_word[i+2])] -= counter[word]
                        freqs[(merged_word, new_word[i+2])] += counter[word]

                    freqs[(new_word[i], new_word[i+1])] -= counter[word]
                    
                    new_word[i] = merged_word
                    del new_word[i + 1]
                else:
                    i += 1

            if len(new_word)!=len(word):
                counter[tuple(new_word)] = counter[word]
                del counter[word]
       
    vocab = dict(zip(vocab.values(), vocab.keys()))
    return vocab, merges