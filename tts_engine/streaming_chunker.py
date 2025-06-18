# streaming_chunker.py

import re

class StreamingChunker:
    def __init__(self, min_words=20):
        if min_words <= 0:
            raise ValueError("min_words must be positive.")
        self.min_words = min_words
        self.potential_boundary_re = re.compile(r'[\.?!]+')
        self.emote_whitelist = {'<chuckle>', '<laugh>', '<sigh>'}
        self.laughter_re = re.compile(r'\b(haha|hehe|hihi)\b', re.IGNORECASE)
        self.action_re = re.compile(r'\*[^\*]+\*|_[^_]+_|\([^)]+\)|[\U0001F300-\U0001F9FF]')
        self.emote_re = re.compile(r'<.*?>')
        self.llm_special_tokens_re = re.compile(r'<\|.*?\|>')
        self.punctuation_space_re = re.compile(r'\s+([.?!,])')

    def _count_words(self, text):
        return len(text.strip().split())

    def _sanitize_chunk(self, text):
        sanitized_text = self.llm_special_tokens_re.sub("", text)

        def emote_replacer(match):
            tag_content = match.group(0).lower()
            if tag_content in self.emote_whitelist:
                return match.group(0)
            return ""
        sanitized_text = self.emote_re.sub(emote_replacer, sanitized_text)
        sanitized_text = self.action_re.sub("", sanitized_text)
        sanitized_text = self.laughter_re.sub("", sanitized_text)
        sanitized_text = sanitized_text.replace("<|", "").replace("|>", "")
        sanitized_text = self.punctuation_space_re.sub(r'\1', sanitized_text)
        return " ".join(sanitized_text.split())

    def _find_first_valid_sentence_end(self, text, is_stream_finished=False):
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
    tokens = [
        "That's a great idea! 😊 ", "Let's plan it for this weekend.", " I was thinking...",
        " maybe we could go hiking?", " *smiles*", " It's been a while since we did that. ",
        "Or, if you prefer, (and I mean this sincerely) we could just relax at home. ",
        "No pressure either way hehe <laugh>. What do you think?",
        "<|endoftext|>"
    ]
    min_words_test_val = 15
    print("--- Test Case: Streaming Chunker with Full Sanitation ---")
    print(f"Parameters: min_words={min_words_test_val}\n")
    expected_chunks = [
        "That's a great idea! Let's plan it for this weekend. I was thinking... maybe we could go hiking?",
        "It's been a while since we did that. Or, if you prefer, we could just relax at home.",
        "No pressure either way <laugh>. What do you think?"
    ]
    chunker_instance = StreamingChunker(min_words=min_words_test_val)
    emitted_chunks   = list(chunker_instance.chunker(iter(tokens)))
    print("Emitted Chunks:")
    for i, chunk in enumerate(emitted_chunks):
        print(f"  Chunk {i+1}: {chunk}")
    print("\nValidation:")
    if emitted_chunks == expected_chunks:
        print("✅ Test Passed: Emitted chunks match expected output exactly.")
    else:
        print("❌ Test Failed: Emitted chunks do not match expected output.")
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