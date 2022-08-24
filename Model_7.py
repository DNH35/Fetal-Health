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
X = fetal_health_drop.drop(["fetal_health"], axis = 1)
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
    tf.keras.layers.Dense(8, activation = 'tanh'),
    tf.keras.layers.Dense(4, activation = 'tanh'),
    tf.keras.layers.Dense(4, activation = 'softmax')
])

fetal_health_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                           optimizer = 'Adam',
                           metrics = ['accuracy'])

history = fetal_health_model.fit(X_train_norm,
                       y_train,
                       epochs = 100,
                       callbacks = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-5 * 10**(epochs/20)),
                       validation_data = (X_test_norm, y_test),
                       verbose = 0)

model_loss = pd.DataFrame(history.history['loss'])
learning_rate = 1e-5 * 10**(tf.range(100)/20)
plt.semilogx(learning_rate, history.history['loss'])
plt.xlabel('log(learning_rate)')
plt.ylabel('Loss')
plt.show()

#print(X_train_norm[0].shape)
tf.random.set_seed(42)
fetal_health_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation = 'tanh'),
    tf.keras.layers.Dense(4, activation = 'tanh'),
    tf.keras.layers.Dense(4, activation = 'softmax')
])

fetal_health_model_2.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                           optimizer = tf.keras.optimizers.Adam(lr = 0.01),
                           metrics = ['accuracy'])

history_2 = fetal_health_model_2.fit(X_train_norm,
                       y_train,
                       epochs = 40,
                       validation_data = (X_test_norm, y_test),
                       verbose = 1)

min_loss_lr = learning_rate[tf.argmin(history.history['loss'])]

tf.random.set_seed(42)
#print(X_train_norm[0].shape)
fetal_health_model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation = 'tanh'),
    tf.keras.layers.Dense(4, activation = 'tanh'),
    tf.keras.layers.Dense(4, activation = 'softmax')
])

fetal_health_model_3.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                           optimizer = tf.keras.optimizers.Adam(lr = 0.03),
                           metrics = ['accuracy'])

history_3 = fetal_health_model_3.fit(X_train_norm,
                       y_train,
                       epochs = 40,
                       validation_data = (X_test_norm, y_test),
                       verbose = 0)

fetal_health_model_2.evaluate(X_test_norm, y_test)
fetal_health_model_3.evaluate(X_test_norm, y_test)

fetal_health_model_2.save('Model_7.h5')