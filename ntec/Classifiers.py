import json
import os
import re
import numpy as np
import pandas as pd
from keras.models import model_from_json

class Classifier():
    """
    Load a ethnicity classifier and its properties.
    """
    def __init__(self, classifier_name):
        self.classifier_name = classifier_name
        self.model_path = os.path.join(os.path.dirname(__file__), "models", classifier_name)
        self.params = self.load_model_params()
        self.classes = self.load_origin_classes()
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the desired classification model.

        Returns:
        --------
        keras object:
            A keras model that can be used for classification.
        """
        # load model architecture:
        arch_file = os.path.join(self.model_path, self.classifier_name + "_architecture.json")
        with open(arch_file) as f:
            ma = json.load(f)
        # build the model:
        model = model_from_json(ma)
        # load and add the trained weights:
        weights_file = os.path.join(self.model_path, self.classifier_name + "_weights.h5")
        model.load_weights(weights_file)
        return model

    def load_model_params(self):
        """
        Loads the parameters of the chosen classifier.

        Returns:
        --------
        dict:
            A dictionary consisting of keys representing classifier parameters, and values representing the values of the chosen classifier.
        """
        param_file = os.path.join(self.model_path, self.classifier_name + "_params.json")
        with open(param_file) as f:
            params = json.load(f)
        return params

    def load_origin_classes(self):
        """
        Loads the ethnic origin classes of the chosen classifier.

        Returns
        -------
        dict:
            A dictionary consisting of keys representing indexes, and values representing class labels the chosen classifier recognizes.
        """
        class_file = os.path.join(self.model_path, self.classifier_name + "_classes.json")
        with open(class_file) as f:
            classes = json.load(f)
        return classes

    def create_padding_matrix(self, name_length: int):
        """
        Creates a padding matrix for a a name given a certain `name_length`.
        
        Parameters
        ----------
        name_length: int
            An integer value indicating the length of the name that has to be enriched with padding instances.

        Returns
        -------
        np.array:
            An 2D-np.array with dimensions that correspond to the classifier's parameters. 
            The array has `shape = (classifier.params["seq_max"] - name_length, classifier.params["n_chars"] + 1)`
        
        See Also
        --------
        Classifier.encode_name : Encodes a name to a 2D-tensor based on the parameters of the chosen classifier.

        Notes
        -----
        Used for padding in `ntec.Classifier.encode_name` method.

        Examples
        --------
        >>> create_padding_matrix(28)
        array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
            ])
        """
        assert isinstance(name_length, int), "`name_length` must be of type `int`."
        n_padding = self.params["seq_max"] - name_length
        padding_matrix = np.zeros(shape=(n_padding, self.params["n_chars"]))
        padding_matrix = np.concatenate([padding_matrix, np.ones(shape = (n_padding, 1))], axis = 1)
        return padding_matrix

    def encode_name(self, name: str):
        """
        Encodes a name to a 2D-tensor based on the parameters of the chosen classifier.

        Parameters
        ----------
        name: str
            A string indicating a name. If the name's length `len(name)` surpasses the `Classifier.params["seq_max]"` parameter,
            only the letters up to `Classifier.params["seq_max]` will be considered.
    
        Returns
        -------
        np.array:
            An np.array of `shape = (classifier.params["seq_max"]`, `classifier.params["n_chars"] + 1)` that represents the one-hot-encoded letters of `name`.

        See Also
        --------
        predict_origins : Predicts the ethnic origin of names.
        create_padding_matrix : Creates a padding matrix for a a name given a certain `name_length`.

        Examples
        --------
        >>> # encoding vectors of the first two letters for the name 'mickey mouse'
        >>> classifier.encode_name("mickey mouse")[:2, :]
        array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ])
        """
        if not isinstance(name, str):
            raise TypeError("Input to argument `name` is not of type 'str'.".format(name))
        assert len(re.findall(r'[^a-z]', "".join(name.split(" ")))) == 0, "`name` contains letters outside the scope of ascii.lowercase. Consider removing using ntec.clean_name()"
        # encode the names letters and add padding dimension
        name_length = len(name)
        if name_length > self.params["seq_max"]:
            name = name[:self.params["seq_max"]]
            name_length = len(name)
        name_encoded = np.array([[0 if c != char else 1 for c in self.params["char_dict"]] + [0] for char in name])
        # fill the padding dimension if necessary
        if name_length < self.params["seq_max"]:
            padding_matrix = self.create_padding_matrix(name_length = name_length)
            name_encoded = np.concatenate([name_encoded, padding_matrix], axis = 0)
        return name_encoded

    def predict_origins(self, x: np.array, output: str = "both"):
        """
        Predicts the ethnic origin of names.

        Parameters
        ----------
        x: np.array
            A np.array of `N` encoded names that are sent to the classifier. The array's shape must correspond to the classifier's parameters.
            The np.array can have `shape = (N, classifier.params["seq_max"], classifier.params["n_chars"] + 1)` (for a sequence of names) or 
            `shape = (classifier.params["seq_max"], classifier.params["n_chars"] + 1)` in which case and additional dimension is added at axis 0 of the array.
        output: str
            A string indicating what values should be returned. Can either be 'classes' (the predicted class labels), 
            'probas' (the predicted class probabilities) or 'both' (perdicted class labels and class probabilities).

        Returns
        -------
        pd.DataFrame:
            A pd.DataFrame containing the predicted class probabilities, the predictet class label(s) or both.
        
        See Also
        --------
        encode_name : Encodes a name to a 2D-tensor based on the parameters of the chosen classifier.
        create_padding_matrix : Creates a padding matrix for a a name given a certain `name_length`.

        Examples
        --------
        >>> name = 'oliver kahn'
        >>> encoded_name = Classifier.encode_name(name = name)
        >>> Classifier.predict_origins(x = encoded_name, output = "classes")
            ethnic_origin
        0       German
        >>> Classifier.predict_origins(x = encoded_name, output = "probas").iloc[:,:7]
        >>> # first seven predicted origin probabilities (alphabetically orderd)
            AngloSaxon    Arabic   Balkans   Chinese  East-European    French    German
        0    0.001329  0.000011  0.000304  0.000009       0.000555  0.002297  0.990167
        """
        # check type and dimensions of the input
        if type(x).__module__ != np.__name__:
            raise TypeError("Input to parameter `x` is not of type 'numpy.ndarray'.")
        if x.ndim != 3:
            x = np.expand_dims(x, 0)
        assert x.shape == (len(x), self.params["seq_max"], self.params["n_chars"] + 1), "Shape of `x` does not correspond to classifier parameters"
        # predict ethnic origin and define output
        probas = self.model.predict(x)
        if output == "probas":
            res = pd.DataFrame(probas, columns=self.classes.values())
        elif output == "classes":
            classes = np.argmax(probas, axis = 1)
            res = pd.Series([self.classes[str(c)] for c in classes], name = "ethnic_origin")
            res = pd.DataFrame(res)
        else:
            res = pd.Series([self.classes[str(c)] for c in np.argmax(probas, axis = 1)], name = "ethnic_origin")
            res = pd.concat([res, pd.DataFrame(probas, columns=self.classes.values())], axis = 1)
        return res
    


        
