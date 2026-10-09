import pytest

from tani_predictor.config import DEFAULT_VOCAB_PATH
from tani_predictor.tokenizer import TaniTokenizer


@pytest.fixture(scope="session")
def tokenizer() -> TaniTokenizer:
    return TaniTokenizer(DEFAULT_VOCAB_PATH)
