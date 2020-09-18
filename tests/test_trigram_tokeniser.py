# *****************************************************************************
# * Copyright 2020 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
from unittest import TestCase

from src.trigram_tokeniser import TrigramTokeniser


class TestTrigramTokeniser(TestCase):
    def test_tokenise(self):
        # Arrange
        input = "Tests are useful but "
        expected = ["Tests are useful", "are useful but"]
        sut = TrigramTokeniser()

        # Act
        actual = sut.tokenise(input)

        self.assertSequenceEqual(expected, actual)
