import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
#Create multiclassification neural network model
tf.random.set_seed(42)
#print(X_train_norm[0].shape)
fetal_health_model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(4, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'softmax')
])

fetal_health_model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
                           optimizer = 'Adam',
                           metrics = ['accuracy'])

fetal_health_model.fit(X_train_norm,
                       tf.one_hot(y_train, depth = 3),
                       epochs = 100,
                       callbacks = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 5),
                       validation_data = (X_test_norm, tf.one_hot(y_test, depth = 3)))

fetal_health_model.save("Model_1.h5")