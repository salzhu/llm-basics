 
import regex as re 
import os 
from tqdm import tqdm

def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]  # Chunks start on previous index, don't include last index
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
            if mini_chunk == b"":  # If EOF, this boundary should be at the end of the file
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)  # Find the special token in the mini chunk
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_worker(input_path, start, end, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        escaped_special_tokens = [re.escape(st) for st in special_tokens]
        pattern_special_tokens = "|".join(escaped_special_tokens)
        chunks = re.split(pattern_special_tokens, chunk)

    pretoken_counts = {}
    for cur_chunk in tqdm(chunks):
        pretoken_iterator = re.finditer(PAT, cur_chunk)
        for pt in pretoken_iterator: 
            token = pt.group()
            if token not in pretoken_counts:
                pretoken_counts[token] = 0
            pretoken_counts[token] += 1

    return pretoken_counts
