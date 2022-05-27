import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.utils import to_categorical

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('hackathon.json').read())
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']
for intent in intents['intents']:
  for pattern in intent['patterns']:
    word_list = nltk.word_tokenize(pattern)
    words.extend(word_list)
    documents.append((word_list, intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
print(words)
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))
training = []
output_empty = [0] * len(classes)

for document in documents:
  bag = []
  word_patterns = document[0]
  word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
  for word in words:
    bag.append(1) if word in word_patterns else bag.append(0)

  output_row = list(output_empty)
  output_row[classes.index(document[1])] = 1
  training.append([bag, output_row])

random.shuffle(training)
training  = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])

train_x = np.array(train_x)
train_y = np.array(train_y)

model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),),activation = 'relu'))
model.add(Dropout (0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout (0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

from nltk import metrics
sgd = SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9, nesterov =  True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
model.fit((train_x), (train_y), epochs = 200, batch_size =5, verbose=1,)
model.save('chatbot.model')
print('Done')

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer=WordNetLemmatizer()
intents=json.loads(open('hackathon.json').read())
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbot.model')

def clean(sentence):
  sentence_words=nltk.word_tokenize(sentence)
  sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
  return sentence_words

def bagwords(sentences):
  sentence_words= clean(sentences)
  bag=[0]*len(words)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word==w:
        bag[i]=1
  return np.array(bag)

def predict(sentence):
  bow= bagwords(sentence)
  res=model.predict(np.array([bow]))[0]
  ERROR_THRESH=0.25
  results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESH]

  results.sort(key=lambda x: x[1], reverse=True)
  retlist = []
  for r in results:
    retlist.append({'intent': classes[r[0]], 'probability': str(r[1])})
  return retlist


def get_response(intents, intents_json):
  tag = intents[0]['intent']
  print(tag)
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      print(result)
      return result
  
print("Go! Bot is running")

while True:
  message = input("")
  ints = predict(message)
  #print(ints)
  #print(intents)
  res = get_response(ints, intents)
  print(res)