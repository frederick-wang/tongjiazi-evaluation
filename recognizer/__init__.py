from recognizer.BaseRecognizer import BaseRecognizer
from recognizer.BertMLMRecognizer import BertMLMRecognizer
from recognizer.NGramRecognizer import NGramRecognizer


def create_ngram_recognizer(**kwargs) -> NGramRecognizer:
    '''Create NGramRecognizer.
    
    Args:
        model_path (str): Path to the model file.

    Raises:
        ValueError: If model_path is not provided.
        ValueError: If model_path is not str.
        ValueError: If any other keyword argument is provided.

    Returns:
        NGramRecognizer: The created NGramRecognizer.
    '''
    if 'model_path' not in kwargs:
        raise ValueError('Missing model_path, required by NGramRecognizer.')
    if type(kwargs['model_path']) is not str:
        raise ValueError(f"Invalid model_path, should be str, got {type(kwargs['model_path'])}.")
    legal_kwargs = {'model_path'}
    for key in kwargs:
        if key not in legal_kwargs:
            raise ValueError(f'Invalid keyword argument: {key}, expected one of {", ".join(legal_kwargs)}')
    model_path = kwargs['model_path']
    return NGramRecognizer(model_path=model_path)


def create_bert_mlm_recognizer(**kwargs) -> BertMLMRecognizer:
    ''' Create BertMLMRecognizer.
    
    Args:
        model_path (str): Path to the model directory.
        device (str): Device to use, 'cpu' or 'cuda'. If not provided, will use 'cuda' if available, otherwise 'cpu'.

    Raises:
        ValueError: If model_path is not provided.
        ValueError: If model_path is not str.
        ValueError: If device is not provided.
        ValueError: If device is not int.
        ValueError: If any other keyword argument is provided.

    Returns:
        BertMLMRecognizer: The created BertMLMRecognizer.
    '''
    if 'model_path' not in kwargs:
        raise ValueError('Missing model_path, required by BertMLMRecognizer.')
    if type(kwargs['model_path']) is not str:
        raise ValueError(f"Invalid model_path, should be str, got {type(kwargs['model_path'])}.")
    if 'device' in kwargs and type(kwargs['device']) is not str:
        raise ValueError(f"Invalid device, should be str, got {type(kwargs['device'])}.")
    legal_kwargs = {'model_path', 'device'}
    for key in kwargs:
        if key not in legal_kwargs:
            raise ValueError(f'Invalid keyword argument: {key}, expected one of {", ".join(legal_kwargs)}')
    model_path = kwargs['model_path']
    device = kwargs.get('device')
    return BertMLMRecognizer(model_path=model_path, device=device)


recognizer_names = ['ngram', 'bert_mlm']


def create_recognizer(name: str, **kwargs) -> BaseRecognizer:
    if name not in recognizer_names:
        raise ValueError(f'Invalid recognizer name: {name}')
    return globals()[f'create_{name}_recognizer'](**kwargs)
