
import spacy
import pytest


@pytest.fixture(scope="session")
def spacy_en():
    yield spacy.load("en_core_web_sm")
