import string
import unidecode
import re

def clean_name(name, ignore_nonLatin = True):
    """
    Cleans a name of digits and punctuation, and turns non-latin letters into Latin counterparts.

    Parameters:
    ----------
    name: str
        A string indicating a name to be cleaned.
    ignore_nonLatin: bool
        A flag indicating if remaining non-Latin letters should be ignored. Default is True.
        If set to False, an Assertation error will be raised if a name still contains non-Latin
        letters after cleaning process.
    
    Returns:
    -------
    str:
        A cleaned version of the original string `name`.
    """
    # check that input is of type 'str':
    if not isinstance(name, str):
        raise TypeError("Input to argument `name` is not of type 'str'.".format(name)) 
    # remove digits and punctuation:
    clean_name = "".join([c for c in name if c not in string.digits and c not in string.punctuation]).lower()
    # transform non-Latin letters: 
    clean_name = unidecode.unidecode(clean_name).strip()
    # check if there are remaining non-latin letters:
    if len(re.findall(r'[^a-z]', "".join(clean_name.split()))) > 0:
        clean_name = "".join([c for c in clean_name if c not in string.digits and c not in string.punctuation]).lower().strip()
    # if remaining non-latin letters should be ignored, remove them from the clean_name
    if ignore_nonLatin:
        clean_name = "".join([c for c in clean_name if c in string.ascii_lowercase + " "])
    # otherwise throw an error if clean_name still has non-latin characters
    assert len(re.findall(r'[^a-z]', "".join(clean_name.split()))) == 0, "non-Latin letters present in the name %s. Transformed to: %s" % (name, clean_name)

    return clean_name