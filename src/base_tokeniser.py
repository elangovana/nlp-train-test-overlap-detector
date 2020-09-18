from typing import List


class BaseTokeniser:
    """
    Base class for tokenising
    """

    def tokenise(self, text: str) -> List[str]:
        raise NotImplementedError()
