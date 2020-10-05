from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from similarity.similarity_splitter import SimilaritySplitter


class TestSimilaritySplitter(TestCase):
    def test_get_case_when_no_threshold(self):
        # Arrange
        ngram = 1
        column = "text"
        mock_comparer = MagicMock()
        mock_comparer.compare.return_value = {
            column: {
                "score": [1.0, 0.0],
                "details": [
                    ("Test1", "Test1")
                    , ("Test5", "Test2")
                ]
            }
        }
        sut = SimilaritySplitter(column=column, ngram=ngram, comparer=mock_comparer)
        test_df = pd.DataFrame([
            {
                "docid": "1",
                column: "Test1"
            }, {
                "docid": "2",
                column: "Test5"
            }])

        train_df = pd.DataFrame([
            {
                "docid": "4",
                column: "Test1"
            },
            {
                "docid": "5",
                column: "Test2"
            }])

        # Act
        actual = sut.get(test_df, train_df, similarity_threshold_min=0, similarity_theshold_max=1.1)

        # Assert
        pd.testing.assert_frame_equal(test_df, actual)

    def test_get_case_when_threshold(self):
        # Arrange
        ngram = 1
        column = "text"
        mock_comparer = MagicMock()
        mock_comparer.compare.return_value = {
            column: {
                "score": [0.8, 0.1],
                "details": [
                    ("Test1", "Test1")
                    , ("Test5", "Test2")
                ]
            }
        }
        sut = SimilaritySplitter(column=column, ngram=ngram, comparer=mock_comparer)
        test_df = pd.DataFrame([
            {
                column: "Test1"
            }, {
                column: "Test5"
            }])

        train_df = pd.DataFrame([
            {
                column: "Test1"
            },
            {
                column: "Test2"
            }])

        expected_df = pd.DataFrame([
            {
                column: "Test1"
            }])

        # Act
        actual = sut.get(test_df, train_df, similarity_threshold_min=.75, similarity_theshold_max=1.1)

        # Assert
        pd.testing.assert_frame_equal(expected_df, actual)
