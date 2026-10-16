import re

import yolali


def test_version_is_pep440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+([ab]\d+|rc\d+)?", yolali.__version__)
