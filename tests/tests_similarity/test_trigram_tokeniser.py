
from unittest import TestCase

from similarity.trigram_tokeniser import TrigramTokeniser


class TestTrigramTokeniser(TestCase):
    def test_tokenise(self):
        # Arrange
        input = "Tests are useful but "
        expected = ["Tests are useful", "are useful but"]
        sut = TrigramTokeniser()

        # Act
        actual = sut.tokenise(input)

        self.assertSequenceEqual(expected, actual)
