
from unittest import TestCase

from similarity.bigram_tokeniser import BigramTokeniser


class TestBigramTokeniser(TestCase):
    def test_tokenise(self):
        # Arrange
        input = "Tests are useful but "
        expected = ["Tests are", "are useful", "useful but"]
        sut = BigramTokeniser()

        # Act
        actual = sut.tokenise(input)

        self.assertSequenceEqual(expected, actual)
