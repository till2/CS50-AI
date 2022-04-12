- had trouble reading the image files with cv2
- thought it was because of the .ppm datatype
- but i quickly figured out that i need to include the full path to the images, not only the names for the imread-function. that fixed it.

- tf-documentation is really helpful. they show examples and explain arguments well. pytorch lacks a bit in this area.

- played around with different optimizers (Adam, SGD)
- appararently minibatches can only be defined in the model.fit-function so the Adam optimizer stayed

- first models had only around 10% accuracy, too few neurons in the dense layers

Now follows a lot of trial-and-error.
- experiment with hyperparams:

First i tinkered with the Deep Neural Net part:
(Here i had 2 Conv2D and 2 MaxPooling2D Layers)

- more Neurons per Layer
2 Dense Layers with 128 neurons each
333/333 - 1s - loss: 0.0519 - accuracy: 0.4977 - 794ms/epoch - 2ms/step

3 Dense Layers with 128 neurons each
333/333 - 1s - loss: 0.0627 - accuracy: 0.4254 - 790ms/epoch - 2ms/step

4 Dense Layers with 128 neurons each
333/333 - 1s - loss: 0.0552 - accuracy: 0.5214 - 831ms/epoch - 2ms/step

5 Dense Layers with 128 neurons each
333/333 - 1s - loss: 0.1045 - accuracy: 0.0544 - 833ms/epoch - 3ms/step


- more Layers, fewer Neurons per Layer
3 Dense Layers with 64 neurons each
333/333 - 1s - loss: 0.0544 - accuracy: 0.5206 - 774ms/epoch - 2ms/step

4 Dense Layers with 64 neurons each
333/333 - 1s - loss: 0.0593 - accuracy: 0.4488 - 844ms/epoch - 3ms/step

5 Dense Layers with 64 neurons each
333/333 - 1s - loss: 0.0733 - accuracy: 0.2711 - 795ms/epoch - 2ms/step


Then i tried out different hyperparameters for the Convolutions and Pooling-Layers:

- 1x Conv2D(filters=4, kernel_size=2) 1x MaxPooling2D(pool_size=2)
333/333 - 1s - loss: 0.0382 - accuracy: 0.6880 - 648ms/epoch - 2ms/step

Much better than the previous runs.

- with filters=8
333/333 - 1s - loss: 0.0504 - accuracy: 0.5533 - 523ms/epoch - 2ms/step

- then i read https://www.jeremyjordan.me/convnet-architectures/ and tried to mimic the general structural ideas of the VGG-16 Network:

- multiple  2x Conv, 1x MaxPooling per block
- multiple blocks (i tried 4)
- the filters double with every block

first run with the new design:
333/333 - 1s - loss: 0.0381 - accuracy: 0.6709 - 963ms/epoch - 3ms/step

- doubled filters for each Conv2D-layer:
333/333 - 1s - loss: 0.0769 - accuracy: 0.3636 - 1s/epoch - 4ms/step

- did some more tests that got worse results (in the 5-20% range)

- reduced it to 2 blocks and only 1 fully connected layer at the end:
333/333 - 1s - loss: 0.0237 - accuracy: 0.8270 - 879ms/epoch - 3ms/step (best so far)

- increased it to 3 blocks
333/333 - 1s - loss: 0.0411 - accuracy: 0.6503 - 903ms/epoch - 3ms/step (got worse)

- back to 2 blocks. first i doubled the number of filters
333/333 - 1s - loss: 0.0284 - accuracy: 0.7915 - 945ms/epoch - 3ms/step (still worse)

back to a reduced number of filters.

- reduced the num of neurons in the fc-layer (fc = fully connected)
333/333 - 1s - loss: 0.0366 - accuracy: 0.7218 - 864ms/epoch - 3ms/step

- added a 2nd fc-layer
333/333 - 1s - loss: 0.0424 - accuracy: 0.6445 - 914ms/epoch - 3ms/step

- back to 1 fc-layer. increased the num of neurons.
333/333 - 1s - loss: 0.0252 - accuracy: 0.8132 - 880ms/epoch - 3ms/step (not better.)

After a couple of hours i feel like an RL agent trying to get at least some rewards (good accuracy).. 

- increased dropout to 0.4:
333/333 - 1s - loss: 0.0327 - accuracy: 0.7627 - 926ms/epoch - 3ms/step

- dropout 0.3:
333/333 - 1s - loss: 0.0329 - accuracy: 0.7746 - 872ms/epoch - 3ms/step

- dropout 0.2:
333/333 - 1s - loss: 0.0170 - accuracy: 0.8980 - 869ms/epoch - 3ms/step (NEW BEST!)

- dropout 0.15:
333/333 - 1s - loss: 0.0251 - accuracy: 0.8351 - 891ms/epoch - 3ms/step (worse)

- without dropout:
333/333 - 1s - loss: 0.0247 - accuracy: 0.8332 - 921ms/epoch - 3ms/step

- with 1x BatchNormalization instead of dropout:
333/333 - 1s - loss: 0.0138 - accuracy: 0.9308 - 991ms/epoch - 3ms/step (YESSS..)

- more BatchNormalization layers between Conv2D layers
333/333 - 1s - loss: 0.0100 - accuracy: 0.9601 - 991ms/epoch - 3ms/step (getting better..)

- even more BatchNormalization:
333/333 - 1s - loss: 0.0149 - accuracy: 0.9237 - 1s/epoch - 3ms/step (nope..)

- one more block:
333/333 - 1s - loss: 0.0110 - accuracy: 0.9336 - 1s/epoch - 3ms/step

- revert.

- reduce neurons in the fc-layer:
333/333 - 1s - loss: 0.0147 - accuracy: 0.9241 - 1s/epoch - 3ms/step


- tried CategoricalCrossentropy loss:
333/333 - 1s - loss: 0.3263 - accuracy: 0.9115 - 1s/epoch - 3ms/step

- tried other losses too they all performed worse than BinaryCrossentropy.

- used LeakyReLU instead of ReLU (alpha=0.05):

- LeakyReLU (alpha=0.07)
333/333 - 1s - loss: 0.0095 - accuracy: 0.9550 - 1s/epoch - 3ms/step

- LeakyReLU(alpha=0.15)
333/333 - 1s - loss: 0.0086 - accuracy: 0.9667 - 1s/epoch - 3ms/step

- LeakyReLU(alpha=0.4)
333/333 - 1s - loss: 0.0089 - accuracy: 0.9694 - 1s/epoch - 3ms/step

LeakyReLU(alpha=0.7)
333/333 - 1s - loss: 0.0115 - accuracy: 0.9525 - 1s/epoch - 3ms/step

LeakyReLU(alpha=0.3)
333/333 - 1s - loss: 0.0100 - accuracy: 0.9564 - 1s/epoch - 3ms/step

LeakyReLU(alpha=0.2)
333/333 - 1s - loss: 0.0094 - accuracy: 0.9596 - 1s/epoch - 3ms/step

- for the final version i used LeakyReLU(alpha=0.15).

- Final model:

**************************************************************************
** 333/333 - 1s - loss: 0.0095 - accuracy: 0.9637 - 1s/epoch - 3ms/step **
**************************************************************************

My general strategy was that if some model architectures had approx. the same
performance, then i would use the one with less parameters to train.
(Saves amount of computation needed and time).

Closing all 50 tabs now and that's it. :)

- tried tips from various youtube videos and from http://d2l.ai/chapter_convolutional-modern/alexnet.html
but could not improve the accuracy further.