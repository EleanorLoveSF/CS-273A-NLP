import data_loader as dl
import os
import random
import numpy as np
import re

train_texts, train_labels, valid_texts, valid_labels = dl.load_train_data('aclImdb/train/')
test_texts, test_labels = dl.load_test_data('aclImdb/test/')

### data cleaning

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

train_texts_clean = preprocess_reviews(train_texts)
valid_texts_clean = preprocess_reviews(valid_texts)
