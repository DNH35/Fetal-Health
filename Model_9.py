import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np
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
X = fetal_health_drop.drop(["fetal_health", 'histogram_tendency'], axis = 1)
y = fetal_health_drop["fetal_health"]

#Create training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
classes_names = ["Normal", "Suspect", "Pathological"]

#Normalizing data
X_train_norm = MinMaxScaler().fit_transform(X_train)
X_test_norm = MinMaxScaler().fit_transform(X_test)

tf.random.set_seed(42)
fetal_health_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = 'tanh'),
    tf.keras.layers.Dense(5, activation = 'tanh'),
    tf.keras.layers.Dense(4, activation = 'softmax')
])

fetal_health_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                           optimizer = tf.keras.optimizers.Adam(lr = 0.003),
                           metrics = ['accuracy'])

history = fetal_health_model.fit(X_train_norm,
                       y_train,
                       epochs = 500,
                       validation_data = (X_test_norm, y_test),
                       verbose = 1)
fetal_health_model.save('Model_9.h5')

model_9_pred = fetal_health_model.predict(X_test_norm)

print(classification_report(y_test, tf.argmax(model_9_pred, axis = 1), target_names=classes_names))
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

create_confusion_matrix(y_test, tf.argmax(model_9_pred, axis = 1), classes = classes_names, figsize = (15, 15), text_size = 15)
plt.show()

