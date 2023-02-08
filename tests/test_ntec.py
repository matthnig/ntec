import pytest
from tests import helpers

# ntec classes and functions to test;
from ntec.clean_name import clean_name
from ntec.Classifiers import Classifier

# model to test
CLASSIFIER = "joeg"

# fixtures -----------
@pytest.fixture
def simulate_name_fixture():
    name = helpers.simulate_name()
    return name

@pytest.fixture
def set_test_classifier(classifier = CLASSIFIER):
    return classifier

@pytest.fixture
def clean_name_fixture(simulate_name_fixture):
    cleaned_name = clean_name(simulate_name_fixture)
    return cleaned_name

@pytest.fixture
def classifier_fixture(set_test_classifier):
    cls = Classifier(classifier_name= set_test_classifier)
    return cls

# tests -----------
def test_clean_name(simulate_name_fixture):
    cleaned_name = clean_name(simulate_name_fixture)
    assert not helpers.non_latin_exist(cleaned_name)

def test_classifier_class(set_test_classifier):
    cls = Classifier(classifier_name= set_test_classifier)
    assert cls
    assert cls.params
    assert len(cls.params) == 3

def test_model_loading(classifier_fixture):
    model = classifier_fixture.load_model() 
    assert model

def test_joeg_encode_name(classifier_fixture, clean_name_fixture):
    encoded_name = classifier_fixture.encode_name(name = clean_name_fixture)
    assert encoded_name.shape == (classifier_fixture.params["seq_max"], classifier_fixture.params["n_chars"] + 1)
    for i in [0, 4, len(clean_name_fixture)-1]:
        assert helpers.get_nth_char_name(name = clean_name_fixture, n = i) == helpers.get_tensor_char_idx(encoded_name=encoded_name, n = i)

def test_wrong_type_cn():
    wrong_inputs = [True, 1, 1.2, ["ah"]]
    for i in wrong_inputs:
        with pytest.raises(TypeError):
            clean_name(name = i)

def test_wrong_type_en(classifier_fixture):
    wrong_inputs = [True, 1, 1.2, ["ah"]]
    for i in wrong_inputs:
        with pytest.raises(TypeError):
            classifier_fixture.encode_name(name = i)