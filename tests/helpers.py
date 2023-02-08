import string
import random
import re
import numpy as np

def simulate_name():
    """
    Simulates a name consisting of punctuation, digits, ascii-lowercase letters and random letters.
    """
    k = random.randint(1, 3)
    punct = random.choices(string.punctuation + " ", k=k)
    digits = [x for x in random.sample(range(10), k=k)]
    ascii_letters = random.choices(string.ascii_letters, k = round(k * 10 * 0.75))
    random_letters = [chr(x) for x in random.sample(range(1000), round(k * 10 * 0.25))]
    
    random_name = punct + digits + ascii_letters + random_letters
    random_name = random.sample(random_name, len(random_name))
    random_name = [str(x) for x in random_name]
    random_name = "".join(random_name)  
    
    return random_name

def non_latin_exist(name):
    """
    Checks if a name-string contains non-ascii lowercase letters.
    """
    assert isinstance(name, str)
    res = len(re.findall(r'[^a-z]', "".join(name.split(" "))))
    res = res > 0
    return res

def get_nth_char_name(name, n = 0):
    """
    Retrieves the nth letter of a name-string
    """
    assert isinstance(name, str)
    char = [i for i, x in enumerate(list(string.ascii_lowercase)) if x == name[n]]
    return char 

def get_tensor_char_idx(encoded_name, n):
    """
    Retrieves a encoded name's nth letter from its encoding matrix. 
    """
    idx = np.where(encoded_name[n,] == 1)[0][0]
    return idx