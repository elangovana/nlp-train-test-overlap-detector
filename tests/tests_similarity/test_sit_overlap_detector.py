from unittest import TestCase

import numpy as np
import pandas as pd

from similarity.cosine_similarity_comparer import CosineSimilarityComparer
from similarity.overlap_detector import OverlapDetector


class TestSitOverlapDetector(TestCase):
    def test_compare(self):
        """
        Test case compare exactly the same sentence
        :return:
        """
        # Arrange

        comparer = CosineSimilarityComparer(1)
        sut = OverlapDetector(comparer)

        src_df = pd.DataFrame([
            {"text": "Protein kinease phosphorylates ps1"}
            , {"text": "The mouse klk3 kinease does not seem to activate ps4"}
        ]
        )
        target_df = pd.DataFrame([
            {"text": "ps1 acteylates ps6"},
            {"text": "Protein kinease phosphorylates ps1"}
            , {"text": "The mouse klk3 kinease does not seem to activate p19"}
        ]
        )
        expected = {
            "text": {
                "score": [100.0, 83.3333],
                "details": [
                    ("Protein kinease phosphorylates ps1",
                     "Protein kinease phosphorylates ps1")
                    , ("The mouse klk3 kinease does not seem to activate ps4",
                       "The mouse klk3 kinease does not seem to activate p19")
                ]
            }
        }

        # Act
        actual = sut.compare(src_df, target_df)

        self.assertSequenceEqual(expected["text"]["score"], np.round(actual["text"]["score"], 4).tolist())
        self.assertSequenceEqual(expected["text"]["details"], actual["text"]["details"])
