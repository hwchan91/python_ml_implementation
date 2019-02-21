from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

# from ch.8
def tokenizer(text):
  text = re.sub('<[^>]*>', '', text) # <(open bracket) (any char except >)* >(close bracket)
  emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) # eyes(:/;/=) optional nose(-) mouth( ) / ( /D/P)
  text = re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
  text = re.sub('\W+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', '') # remove non words(\W), add back emoticons
  tokenized = [w for w in text.split() if w not in stop]
  return tokenized

vect = HashingVectorizer(
  decode_error='ignore',
  n_features=2**21,
  preprocessor=None,
  tokenizer=tokenizer
)
