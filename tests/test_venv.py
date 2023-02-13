"""
Test in_virtualenv
"""

from venv_torch.venv import in_virtualenv


def test_in_virtualenv():
    assert in_virtualenv() == True
