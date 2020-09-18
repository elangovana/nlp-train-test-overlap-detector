
from unittest import TestCase

from src.unigram_tokeniser import UnigramTokeniser


class TestUnigramTokeniser(TestCase):
    def test_tokenise(self):
        # Arrange
        input = "Tests are useful"
        expected = ["Tests", "are", "useful"]
        sut = UnigramTokeniser()

        # Act
        actual = sut.tokenise(input)

        self.assertSequenceEqual(expected, actual)
