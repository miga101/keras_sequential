import numpy
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# if train:
# create the Model
model = Sequential()
# First hidden layer with 8 input units
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
# Second hidden layer with 8 units
model.add(Dense(8, init='uniform', activation='relu'))
# output layer with output 1 unit
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit/train Model
#model.fit(X, Y, nb_epoch=150, batch_size=10)
#model.fit(X, Y, nb_epoch=150, batch_size=10)
history = model.fit(X, Y, validation_split=0.33, nb_epoch=500,
                    batch_size=10, verbose=1)

# # Evaluate Model
# scores = model.evaluate(X, Y)
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# rounded = [round(x) for x in predictions]
# print(rounded)

# check predictions:
for j in range(1):
    print("test group: %d" % j)
    corect = 0.0
    for i in range(len(X)):
        #x = numpy.random.randint(1, len(X))
        x = i
        xOri = Y[x]
        xPre = predictions[x]
        intXpre = round(xPre)
        mark = ' '
        if xOri == intXpre:
            corect += 1.0
            mark = '*'
        accu = corect / (i+1)

        print("%3d ori: %d, pre: %.2f  %d -> %.2f  %s" %(x+1, xOri, xPre, intXpre, accu, mark))
        # xOri = Y[i]
        # xPre = predictions[i]
        # print("%3d ori: %d, pre: %d" % (i+1, xOri, xPre))

    print("accuracy: %.2f" % (accu))
    print("error: %.2f" % (1.0 - accu))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()















