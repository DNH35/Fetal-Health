import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

#Inputing data
fetal_health = pd.read_csv("fetal_health.csv")

#Create correlation between features and labels
fetal_corrmat = fetal_health.corr()

#Create a function that shows correlation between each features and labels
def show_correlation_matrix(data):
    corrmat = data.corr()
    cmap = sns.diverging_palette(205, 133, 63, as_cmap=True)
    cols = (["#B0E0E6", "#87CEFA", "#4682B4", "#CD853F", "#DEB887", "#FAEBD7"])
    f, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(corrmat,cmap=cols,annot=True)
    plt.show()

#Drop data that has correlation with label that is less than |0.1|
delete_list = []
for i, value in enumerate(fetal_corrmat["fetal_health"]):
    if abs(value) < 0.1:
        delete_list.append(i)
fetal_health_drop = fetal_health.drop(fetal_health.columns[delete_list], axis = 1)

#Create features and labels
X = fetal_health_drop.drop("fetal_health", axis = 1)
y = fetal_health_drop["fetal_health"]

#Create training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
classes_names = ["Normal", "Suspect", "Pathological"]

#Normalizing data
X_train_norm = MinMaxScaler().fit_transform(X_train)
X_test_norm = MinMaxScaler().fit_transform(X_test)

#One hot label
y_train_one_hot = tf.one_hot(y_train, depth = 3)
y_test_one_hot = tf.one_hot(y_test, depth = 3)

print(X_train_norm.shape)

model_1 = tf.keras.models.load_model('Model_1.h5')
model_1.evaluate(X_test_norm, y_test_one_hot)

#Model_2 is similar to model_1 but modifying the learning rate to find the optimal learning rate
model_2 = tf.keras.models.load_model('Model_2.h5')
model_2.evaluate(X_test_norm, y_test_one_hot)

#Model_3 increases the number of hidden layers and neurons
model_3 = tf.keras.models.load_model('Model_3.h5')
model_3.evaluate(X_test_norm, y_test_one_hot)

model_4 = tf.keras.models.load_model('Model_4.h5')
model_4.evaluate(X_test_norm, y_test_one_hot)

#Model 5: adding regularizor did not give any better result

#Model 6: changing activation function from relu to tanh, change loss function to sparse
model_6 = tf.keras.models.load_model('Model_6.h5')
model_6.evaluate(X_test_norm, y_test)

#Model 7: investigate optimal learning rate of tanh with sparse
model_7 = tf.keras.models.load_model('Model_7.h5')
model_7.evaluate(X_test_norm, y_test)

#Model 8: use lr from model 7, increase hidden layers and neurons and train for longer time
model_8 = tf.keras.models.load_model('Model_8.h5')
model_8.evaluate(X_test_norm, y_test)

model_8_pred = model_8.predict(X_test_norm)
print(classification_report(y_test, np.argmax(model_8_pred, axis = 1), target_names=classes_names))
def create_confusion_matrix(y_true, y_pred, classes = None, figsize = (10, 10), text_size = 15):
  # Create the confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0]

  # Let's prettify it
  fig, ax = plt.subplots(figsize=figsize)
  # Create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.matshow.html
  fig.colorbar(cax)

#Set labels to classes

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

  # Set x-axis labels to bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Adjust label size
  ax.xaxis.label.set_size(text_size)
  ax.yaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  # Set threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size)

create_confusion_matrix(y_test, tf.argmax(model_8_pred, axis = 1))
plt.show()

#Conclusion: Even though model 8 performs slightly better than model 7, model 7 was able to determine fetal health that is in pathological a lot better
#Model 9, 10, 11 takes out the feature histogram_tendency to remove negative input. Overall, model 9 predicts the best.