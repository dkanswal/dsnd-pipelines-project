import pandas as pd
import numpy as np
import spacy

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/reviews.csv")

df.head()