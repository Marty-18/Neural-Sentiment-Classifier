import pandas as pd
import numpy as np


class SentiData:
    def __init__(self, file, tokenizer, vocab_size=10000, max_len=267):
        self.data = pd.load_csv(file)
        self.tokenizer = tokenizer
