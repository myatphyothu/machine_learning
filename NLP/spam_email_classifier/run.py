import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf

df = pd.read_csv('spam_emails.csv')
print(df.head(5))
