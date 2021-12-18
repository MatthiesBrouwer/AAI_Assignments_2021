"""
MNIST opdracht A: "Only Dense"      (by Marius Versteegen, 2021)
Bij deze opdracht ga je handgeschreven cijfers klassificeren met 
een "dense" network.

De opdracht bestaat uit drie delen: A1, A2 en A3 (zie verderop)

Er is ook een impliciete opdracht die hier niet wordt getoetst
(maar mogelijk wel op het tentamen):
    
--> Zorg ervoor dat je de onderstaande code volledig begrijpt. <--
******************************************************************

Tip: stap in de Debugger door de code, en bestudeer de tussenresultaten.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from MavTools_NN import ViewTools_NN
import tensorflow as tf
#tf.random.set_seed(0) #for reproducability

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(x_test[0])
print("Showing first test-image\n")
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.grid(False)
plt.show()

inputShape = x_test[0].shape

# show the shape of the training set and the amount of samples
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples\n")

# convert class vectors to binary class matrices (one-hot encoding)
# for example 3 becomes (0,0,0,1,0,0,0,0,0,0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# we can flatten the images, because we use Dense layers only
x_train_flat = x_train.reshape(x_train.shape[0], inputShape[0]*inputShape[1])
x_test_flat  = x_test.reshape(x_test.shape[0], inputShape[0]*inputShape[1])

"""
Opdracht A1: 
    
Voeg ALLEEN Dense en/of Dropout layers toe aan het onderstaande model.
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0,95.

Voorbeelden van layers:
    layers.Dropout( getal ),
    layers.Dense(units=getal, activation='sigmoid'),
    layers.Dense(units=getal, activation='relu'),

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
en beargumenteer elk van je stappen.

[ervaring: Zonder GPU, op I5 processor < 5 min, 250 epochs bij batchsize 4096]

Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_A.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.


[ANTWOORD]
Ik ben gekomen op deze configuratie door trial and error.

De eerste dense-layer ben ik op gekomen doordat de output 10 neurons nodig had (een voor iedere digit).
Het leek mij handig als de eerste layer dan een significant grotere hoeveelheid had, die wel tot een 
gelijdelijke groei zou leiden. Daarom leek mij een tienvoud ervan voldoende.

Vervolgens verzon ik de volgende twee layers op hetzelfde principe.
Eerst heb ik op papier alle features uitgeschreven die ikzelf kon onderscheiden:
	- De 8 bestaat uit 2 circles. Beide zin op te delen in 4 kleinere boogjes (dus 8 voor twee circles)
	- De 5, 7 en 2 hebben beide een horizontaal streepje boven en onder (nog 2 features)
	- De 4 en de 1 hebben een verticale streep. (nog 1 feature)
	- De 7 heeft een schuine streep (1 feature)
	- De 6, 9, en 0 hebben wel de circledelen van de 8, maar hebben in het midden geen inkeping.
	  Daarom tel ik hier nog 2 features bij die bogen aan de zijkanten zijn (nog 2 features)
	- Als laatste heeft de 4 een horizontale streep in het midden. (1 feature)

In totaal dus 15 features. Daarom bestaat de hidden layer voorgaand aan de ouput uit 15 neuronen.
Daarnaast heb ik hetzelfde principe toegepast als bij de 100 neuronen in de eerste hidden layers (wat dus een
meervoud was van de laatste hidden layer), alleen dit keer met tweevoud (15 * 2 = 30). 

"""

def buildMyModel(inputShape):
	model = keras.Sequential(
		[
			keras.Input(shape=inputShape),
			layers.Dense(units=100, activation='sigmoid'),
			layers.Dropout(.1),	
			layers.Dense(units=30, activation='sigmoid'),
			layers.Dropout(.1),
			layers.Dense(units=15, activation='sigmoid'),
			layers.Dropout(.1),
			layers.Dense(units=num_classes, activation='sigmoid')
		]
	)
	return model
model = buildMyModel(x_train_flat[0].shape)
model.summary()

"""
Opdracht A2: 

Verklaar met kleine berekeningen het aantal parameters dat bij elke laag
staat genoemd in bovenstaande model summary.


Dense layer 1 (100 neurons):
	Total params: 78500
	Calculations:
		28 * 28 = 784 (pixel amount per image)
		(100 * 784) + 100 = 78500 (weights per neuron per pixel + the bias of each neuron)

Dense layer 2 (30 neurons):
	Total params: 3030
	Calculations:
		30 * 100 = 3000 (Weights per neuron per previous neurons)
		3000 + 30 = 3030 (Add the bias of each neuron)

Dense layer 3 (15 neurons):
	Total params: 465
	Calculations:
		15 * 30 = 450 (Weights per neuron per previous neurons)
		450 + 15 = 465 (Add the bias of each neuron)


Dense layer 4 (10 neurons)(output layer):
	Total params: 160
	Calculations:
		10 * 15 = 150 (Weights per neuron per previous neurons)
		150 + 10 = 160 (Add the bias of each neuron)
"""

"""
# Train the model
"""
batch_size = 1000   # Larger can mean faster training (especially when using the gpu: 4096), 
                  # but requires more system memory. Select it properly for your system.
                    
epochs = 100     # it's probably more then you like to wait for,
                  # but you can interrupt training anytime with CTRL+C

learningrate = 0.01
#loss_fn = "categorical_crossentropy" # can only be used, and is effictive for an output array of hot-ones (one dimensional array)
loss_fn = 'mean_squared_error'        # can be used for other output shapes as well. seems to work better for categorical as well..

optimizer = keras.optimizers.Adam(lr=learningrate)
model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("\nx_train_flat.shape:", x_train.shape)
print("y_train.shape", y_train.shape)

bInitialiseWeightsFromFile = False # Set this to false if you've changed your model, or if you want to restart using different (random) weights.
learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01
if(bInitialiseWeightsFromFile):
	model.load_weights("myWeights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_A_weights.h5" here.

#print("Device: {}".format(tf.device_name))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print (ViewTools_NN.getColoredText(255,255,0,"\nJust type CTRL+C anytime if you feel that you've waited for enough episodes.\n"))

try:
	model.fit(x_train_flat, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
	print("interrupted fit by keyboard\n")

"""
# Evaluate the trained model
"""
score = model.evaluate(x_test_flat, y_test, verbose=0)
print("Test loss:", score[0])       # 0.0394
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]), "\n")   # 0.9903

model.summary()
model.save_weights('myWeights.h5')

prediction = model.predict(x_test_flat)
print("\nFirst test sample: predicted output and desired output:")
print(prediction[0])
print(y_test[0],"\n")

# study the meaning of the filtered outputs by comparing them for
# a few samples
nLastLayer = len(model.layers)-1
nLayer = nLastLayer -1                 # this time, I select the last layer, such that the end-outputs are visualised.
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)

"""
Opdracht A3: 

1 Leg uit wat de betekenis is van de output images die bovenstaand genenereerd worden met "printFeatureMapsForLayer".
	Bovenstaande plots geven de 10 output neurons weer. Deze kunnen of een 0 (zwart) of een 1 (wit) geven.
	Vervolgens word dit door het NN gebruikt om aan te geven welk getal geidentificeerd is. Zo is alleen
	het derde vierkantje een output van 1 en is de rest een 0 wannee een "2" is geidentifiseerd (de derde
	aangezien de 0 op de eerste plek staat). 

2 Leg uit wat de betekenis zou zijn van wat je ziet als je aan die functie de index van de eerste dense layer meegeeft.
	Wanneer de eerste dense layer word meegegeven zullen er enorm veel vierkantjes worden weergegeven. 
	Het netwerk gebruikt deze layer hoogst waarschijnlijk voor de eerste feature onderscheidingen. Latere layers kunnen deze
	vervolgens gebruiken om complexere features te onderscheiden. De output hiervan zal enorm basaal zijn en ik verwacht ook 
	dat het veel last zal hebben van overfitting aangezien er zoveel stijlen van handwriting zijn.

3 Leg uit wat de betekenis zou zijn van wat je ziet als je aan die functie de index van de tweede dense layer meegeeft.
	Zoals bovenstaand beschreven hebben latere layers een vervolg op de eerdere. De tweede layer zal daarom hoogstwaarschijnlijk 
	de gevonden features in de vorige layer combineren om complexere features te kunnen identificeren.
"""
