"""
MNIST opdracht C: "Only Conv"      (by Marius Versteegen, 2021)

Bij deze opdracht gebruik je geen dense layer meer.
De output is nu niet meer een vector van 10, maar een
plaatje van 1 pixel groot en 10 lagen diep.

Deze opdracht bestaat uit vier delen: C1 tm C4 (zie verderop)
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import random
from MavTools_NN import ViewTools_NN

random.seed(0)
np.random.seed(0)
# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(x_test[0])

print("show image\n")
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Conv layers expect images.
# Make sure the images have shape (28, 28, 1). 
# (for RGB images, the last parameter would have been 3)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# change shape 60000,10 into 60000,1,1,10  which is 10 layers of 1x1 pix images, which I use for categorical classification.
y_train = np.expand_dims(np.expand_dims(y_train,-2),-2)
y_test = np.expand_dims(np.expand_dims(y_test,-2),-2)

"""
Opdracht C1: 
    
Voeg ALLEEN Convolution en/of MaxPooling2D layers toe aan het onderstaande model.
(dus GEEN dense layers, ook niet voor de output layer)
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0.98.

Voorbeelden van layers:
    layers.Conv2D(getal, kernel_size=(getal, getal))
    layers.MaxPooling2D(pool_size=(getal, getal))

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
en beargumenteer elk van je stappen.

BELANGRIJK (ivm opdracht D, hierna):  
* Zorg er dit keer voor dat de output van je laatste layer bestaat uit een 1x1 image met 10 lagen.
Met andere woorden: zorg ervoor dat de output shape van de laatste layer gelijk is aan (1,1,10)
De eerste laag moet 1 worden bij het cijfer 0, de tweede bij het cijfer 1, etc.

Tip: Het zou kunnen dat je resultaat bij opdracht B al aardig kunt hergebruiken,
     als je flatten en dense door een conv2D vervangt.
     Om precies op 1x1 output uit te komen kun je puzzelen met de padding, 
     de conv kernal size en de pooling.
     
* backup eventueel na de finale succesvolle training van je model de gegenereerde weights file
  (myWeights.m5). Die kun je dan in opdracht D inladen voor snellere training.
  
  
Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_C.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.


[ANTWOORD]
Wederom maak ik gebruik van vermenigvuldigingen van de 15 features die ik voor
opdracht A had verzonnen. Dit zijn de kleinst mogelijke feature onderdelen 
van de mogelijk getallen. Door vermenigvuldigingen te gebruiken sta ik het NN
toe om overzichtelijk features te kunnen combineren tot complexere versies. 

Daarom maak ik gebruik van 3 layers, waarvan ieder een vermenigvuldiging van 15 is.

Ook ben ik gegaan voor een basis pooling van (2,2) zodat het net zoals de vermenigvuldigingen
geleidelijk kan werken.

"""

def buildMyModel(inputShape):
	model = keras.Sequential(
		[
			keras.Input(shape=inputShape),
			layers.Conv2D(45, kernel_size=(3, 3)),
			layers.MaxPooling2D(pool_size=(2, 2)),

			layers.Conv2D(30, kernel_size=(3, 3)),	
			layers.MaxPooling2D(pool_size=(2, 2)),

			layers.Conv2D(15, kernel_size=(3, 3)),	
			layers.MaxPooling2D(pool_size=(2, 2)),

			layers.Conv2D(10, kernel_size=(1, 1))

		]
	)
	return model

model = buildMyModel(x_train[0].shape)
model.summary()

"""
Opdracht C2: 
    
Kopieer bovenstaande model summary en verklaar bij 
bovenstaande model summary bij elke laag met kleine berekeningen 
de Output Shape


Test loss: 0.010030196979641914
Test accuracy: 0.9818999767303467 
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 45)        450       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 45)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 30)        12180     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 5, 5, 30)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 3, 3, 15)          4065      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 1, 1, 15)         0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 1, 1, 10)          160       
                                                                 
=================================================================
Total params: 16,855
Trainable params: 16,855
Non-trainable params: 0


Convolution layer 1 (45 layers):
	Total params: 450
	Calculations:
		3 * 3 * 45 = 405 (De hoeveelheid gewichten (3*3) vermenigvuldigt met het aantal layers)
		405 + 45 = 450 (Voeg biases toe)	

	Output Shape: (None, 26, 26, 45)
		De (26, 26) komt doordat er een 3x3 filter over een matrix (de image) van 28x28 word gehaald
		zonder de "same" padding. Dit zorgt ervoor dat er van beide axes (horizontaal en verticaal) 
		aan de zijkanten 1 pixelwaarde verdwijnt. Daarom word de nieuwe grootte (26x26)	

		Verder zijn er 45 layers

	Output Shape: (None, 13, 13, 45)
		De (13, 13) shape komt omdat pooling in dit geval pixels in een matrix van (2,2) samenvoegt,
		waardoor in de praktijk de (26,26) matrix (overgebleven na de convolutie) word verkleind tot
		een shape van 2x zo klein.

	Output Shape: (None, 11, 11, 30)
		Er word nog een keer een convolutie matrix over de (13,13) shape (overgebleven na de pooling) zonder
		de "same" padding gehaald. Dus ook deze word met 2 aan beide kanten verkleint.
	
		Verder zijn er 30 layers

	
	Output Shape: (None, 5, 5, 30)
		De (5, 5) shape komt omdat pooling in dit geval pixels in een matrix van (2,2) samenvoegt,
		waardoor in de praktijk de (11, 11) matrix (overgebleven na de convolutie) word verkleind tot
		een shape van 2x zo klein. Omdat 11 niet te delen is door 2 word er naar beneden afgerond en blijft
		er een (5, 5) shape over.

	
	Output Shape: (None, 3, 3, 15)
		Er word nog een keer een convolutie matrix over de (5, 5) shape (overgebleven na de pooling) zonder
		de "same" padding gehaald. Dus ook deze word met 2 aan beide kanten verkleint.
	
		Verder zijn er 15 layers



	Output Shape: (None, 1, 1, 15)
		De (1, 1) shape komt omdat pooling in dit geval pixels in een matrix van (2,2) samenvoegt,
		waardoor in de praktijk de (3, 3) matrix (overgebleven na de convolutie) word verkleind tot
		een shape van 2x zo klein. Dit ko

		Tevens kan de (3,3) layer niet net door 2 worden gedeeld, en rond dus af tot 1

	
	Output Shape: (None, 1, 1, 10)
		Dit is de output layerEr word nog een keer een convolutie matrix over de (1, 1) shape (overgebleven na de pooling).
		Wederom word er geen gebruik gemaakt van de "same" padding, maar omdat de convolutielayer maar (1,1) is valt er 
		niks van de zijkanten af te halen.

		Verder zijn er 10 layers, 1 voor iedere digit

"""

"""
Opdracht C3: 
    
Verklaar nu bij elke laag met kleine berekeningen het aantal parameters.


[ANTWOORD]

Convolution layer 1 (45 layers):
	Total params: 450
	Calculations:
		3 * 3 * 45 = 405 (Het totaal aantal layers vermenigvuldigt met het aantal weights per layer)
		405 + 45 = 450 (Voeg bias toe)

Convolution layer 2 (30 layers):
	Total params: 12180
	Calculation:
		3 * 3 * 30 = 270 (Het totaal aantal layers vermenigvuldigt met het aantal weights per layer)
		270 * 45 = 12150 (Vermenigvuldig met het totaal aantal layers van de vorige convolution layer)
		12150 (Voeg de bias van iedere layer in deze convolution layer toe)

Convolution layer 3 (15 layers):
	Total params: 4065
	Calculations:
		3 * 3 * 15 = 135 (Het totaal aantal layers vermenigvuldigt met het aantal weights per layer)
		135 * 30 = 4050 (Vermenigvuldig met het totaal aantal layers van de vorige convolution layer)
		4050 + 15 = 4065 (Voeg de bias van iedere layer in deze convolution layer toe)

Convolution layer 4 (10 layers):
	Total params: 160
	Calculations
		1 * 1 * 10 = 10 (Het zijn 10 layers met elk maar een 1x1 matrix. Dus maar 10 params)
		10 * 15 = 150 (Vermenigvuldig met het totaal aantal layers van de vorige convolution layer)
		150 + 10 = 160 (Voeg de bias van iedere layer in deze convolution layer toe)


"""

"""
Opdracht C4: 
    
Bij elke conv layer hoort een aantal elementaire operaties (+ en *).
* Geef PER CONV LAYER een berekening van het totaal aantal operaties 
  dat nodig is voor het klassificeren van 1 test-sample.
* Op welk aantal operaties kom je uit voor alle conv layers samen?


[Antwoord]

Een convolution layer met matrices van size (x,y) doet voor iedere layer in de input image
de volgende berekeningen:
	- Kies een pixel met een proximity aan neighbours van size (x,y)
	- Plaats de matrix (metaphorish gezien) op de middelste pixel en verm


Convolution layer 1 (45 layers):
	Input layers: 1 layer (de image zelf)
	Kernel size: 3x3
	Calculations:
		3 * 3 = 9 (De vermenigvuldigingen voor het berekenen van 1 filter uitvoering op een pixel en bijhorende neighbours)
		9 + 8 + 1 = 18 (Het optellen van de waardes(8) plus het optellen van de bias(1))
		18 * 26 * 26 = 




"""

"""
## Train the model
"""

batch_size = 4096 # Larger means faster training, but requires more system memory.
epochs = 1000 # for now

bInitialiseWeightsFromFile = True # Set this to false if you've changed your model.

learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01

# We gebruiken alvast mean_squared_error ipv categorical_crossentropy als loss method,
# omdat straks bij opdracht D ook de afwezigheid van een cijfer een valide mogelijkheid is.
optimizer = keras.optimizers.Adam(lr=learningrate) #lr=0.01 is king
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("x_train.shape")
print(x_train.shape)

print("y_train.shape")
print(y_train.shape)

if (bInitialiseWeightsFromFile):
	model.load_weights("myWeights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_C_weights.h5" here.
try:
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
	print("interrupted fit by keyboard")

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]))

model.summary()

model.save_weights('myWeights.h5')

prediction = model.predict(x_test)
print(prediction[0])
print(y_test[0])

# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)


print(x_test.shape)

# study the meaning of the filtered outputs by comparing them for
# multiple samples
nLastLayer = len(model.layers)-1
nLayer=nLastLayer
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
x_test_flat=None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)
