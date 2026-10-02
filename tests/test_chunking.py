import pytest

from rag_assistant.chunking import chunk_text


def test_chunks_respect_sentence_boundaries():
    text = "First sentence here. Second one follows. Third is last."
    chunks = chunk_text("d", text, max_words=6, overlap=0)
    assert all(c.text.endswith(".") for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_trailing_sentence():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    chunks = chunk_text("d", text, max_words=6, overlap=3)
    assert "Delta epsilon zeta." in chunks[0].text and "Delta epsilon zeta." in chunks[1].text


def test_empty_text_gives_no_chunks():
    assert chunk_text("d", "   ") == []


@pytest.mark.parametrize("max_words,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_raise(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_text("d", "text.", max_words, overlap)
