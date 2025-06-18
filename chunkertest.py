import re

class StreamingChunker:
    """
    A chunker that groups a stream of LLM tokens into natural sentences.
    It accumulates complete sentences until the total word count meets or exceeds
    min_words, at which point it yields the accumulated group as one chunk.
    It correctly handles the final chunk by appending short trailing sentences.
    """

    def __init__(self, min_words=20):
        if min_words <= 0:
            raise ValueError("min_words must be positive.")
        self.min_words = min_words
        self.potential_boundary_re = re.compile(r'[\.?!]+')

        self.emote_whitelist = {'<chuckle>', '<laugh>', '<sigh>'}
        self.laughter_re = re.compile(r'\b(haha|hehe|hihi)\b', re.IGNORECASE)
        self.emphasis_re = re.compile(r'\*.*?\*')
        self.emote_re = re.compile(r'<.*?>')
        self.punctuation_space_re = re.compile(r'\s+([.?!,])')


    def _count_words(self, text):
        """Counts words by stripping the text and splitting on whitespace."""
        return len(text.strip().split())

    def _sanitize_chunk(self, text):
        """
        Sanitizes a chunk of text by removing unwanted artifacts and fixing spacing.
        """
        def emote_replacer(match):
            tag_content = match.group(0).lower()
            if tag_content in self.emote_whitelist:
                return match.group(0)
            return ""

        sanitized_text = self.emote_re.sub(emote_replacer, text)

        sanitized_text = self.emphasis_re.sub("", sanitized_text)

        sanitized_text = self.laughter_re.sub("", sanitized_text)

        sanitized_text = self.punctuation_space_re.sub(r'\1', sanitized_text)

        return " ".join(sanitized_text.split())

    def _find_first_valid_sentence_end(self, text, is_stream_finished=False):
        """
        Finds the end position of the first valid sentence boundary.
        A boundary is invalid if it's immediately followed by a comma.
        A boundary at the end of the buffer is only confirmed if the stream is finished.
        """
        for match in self.potential_boundary_re.finditer(text):
            end_pos = match.end()
            whats_next = text[end_pos:]

            if not whats_next.strip():
                if is_stream_finished:
                    return end_pos
                else:
                    continue

            if whats_next.lstrip().startswith(','):
                continue

            return end_pos
        return -1

    def chunker(self, token_iterator):
        """A generator that takes an iterator of tokens and yields text chunks."""
        current_chunk_sentence_parts = []
        pending_token_accumulator = ""

        tokens = list(token_iterator)

        for i, token in enumerate(tokens):
            is_last_token = (i == len(tokens) - 1)
            pending_token_accumulator += token

            while True:
                boundary_pos = self._find_first_valid_sentence_end(
                    pending_token_accumulator, is_stream_finished=is_last_token
                )

                if boundary_pos == -1:
                    if not is_last_token:
                        break

                if boundary_pos == -1 and is_last_token:
                    completed_sentence_part = pending_token_accumulator
                    pending_token_accumulator = ""
                else:
                    completed_sentence_part = pending_token_accumulator[:boundary_pos]
                    pending_token_accumulator = pending_token_accumulator[boundary_pos:]

                current_chunk_sentence_parts.append(completed_sentence_part)
                potential_chunk = "".join(current_chunk_sentence_parts)

                if self._count_words(potential_chunk) >= self.min_words and not is_last_token:
                    yield self._sanitize_chunk(potential_chunk)
                    current_chunk_sentence_parts = []

                if not pending_token_accumulator.strip():
                    break

        if current_chunk_sentence_parts:
            final_chunk_text = "".join(current_chunk_sentence_parts)
            if final_chunk_text.strip():
                yield self._sanitize_chunk(final_chunk_text)


def test_simplified_chunker():
    """
    Tests the corrected simplified StreamingChunker with a simulated LLM token stream,
    now including tests for the new sanitation logic.
    """
    tokens = [
        "..", "so", " you", " know", " when", " we", " passed", " by", " that", " ..", "um", " ..", "nice", " house", " at", " the", " corner", "..", "?",
        " I", "'ve", " been", " thinking", " haha", " about", " it", " ever", " since", ".",
        " I", " would", " really", " love", " *smiles*", " to", " live", " there", "<random_tag>", "...",
        " not", " to", " uh", ",", " put", " any", " ide", "as", " in", " your", " *shrugs*", " head", "..", ",", " o", "nly", " if", " you", " also", " want", " to", " live", " hehe", " there", " with", " me", "..",
        " and", " if", " we", " can", " afford", " it", " <laugh>", " !"
    ]


    min_words_test_val = 12
    print("--- Test Case: Simplified Chunker (Corrected Logic with Sanitation) ---")
    print(f"Parameters: min_words={min_words_test_val}\n")

    expected_chunks = [
      "..so you know when we passed by that..um..nice house at the corner..?",
      "I've been thinking about it ever since. I would really love to live there...",
      "not to uh, put any ideas in your head.., only if you also want to live there with me..",
      "and if we can afford it <laugh>!"
    ]

    chunker_instance = StreamingChunker(min_words=min_words_test_val)
    emitted_chunks   = list(chunker_instance.chunker(iter(tokens)))

    print("Emitted Chunks:")
    for i, chunk in enumerate(emitted_chunks):
        print(f"  Chunk {i+1}: {chunk}")

    print("\nValidation:")
    if emitted_chunks == expected_chunks:
        print("Test Passed: Emitted chunks match expected output exactly.")
    else:
        print("Test Failed: Emitted chunks do not match expected output.")
        max_len = max(len(emitted_chunks), len(expected_chunks))
        for i in range(max_len):
            emitted = emitted_chunks[i] if i < len(emitted_chunks) else "N/A"
            expected = expected_chunks[i] if i < len(expected_chunks) else "N/A"
            if emitted != expected :
                print(f"  Mismatch at chunk {i+1}:")
                print(f"    Got:      '{emitted}'")
                print(f"    Expected: '{expected}'")

if __name__ == "__main__":
    test_simplified_chunker()