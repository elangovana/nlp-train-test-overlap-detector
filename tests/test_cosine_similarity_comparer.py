from unittest import TestCase
from unittest.mock import MagicMock

from src.cosine_similarity_comparer import CosineSimilarityComparer


class TestCosineSimilarityComparer(TestCase):
    def test__call___same(self):
        """
        Test case compare exactly the same sentence
        :return:
        """
        # Arrange
        tokeniser_mock = MagicMock()
        tokeniser_mock.tokenise = lambda x: x.split(" ")
        sut = CosineSimilarityComparer(tokeniser_mock)
        src_sentence = "This is a sample"
        expected_score = [100]

        # Act
        actual = sut([src_sentence], [src_sentence])

        self.assertEqual(expected_score, actual)

    def test__call___different(self):
        """
        Test case compare completely different sentences
        :return:
        """
        # Arrange
        tokeniser_mock = MagicMock()
        tokeniser_mock.tokenise = lambda x: x.split(" ")
        sut = CosineSimilarityComparer(tokeniser_mock)
        src_sentence = "This is a sample"
        target_sentence = "Magic Mountain is amazing"
        expected_score = [0]

        # Act
        actual = sut([src_sentence], [target_sentence])

        self.assertEqual(expected_score, actual)
