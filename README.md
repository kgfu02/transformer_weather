# transformer_weather
Transformer model made with Tensorflow for predicting temperature from [this](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip) dataset ([credits](https://www.bgc-jena.mpg.de/wetter/)). Model code based off of https://www.tensorflow.org/text/tutorials/transformer and https://www.tensorflow.org/tutorials/structured_data/time_series (converted from NLP to timeseries analysis). 

Some notes:
1. Key changes to Tensorflow tutorial code include: replacing embedding layer with a dense layer, changing dimensions/indexes in function to work with new dataset, MAE loss function. 
2. Trained over 40 epochs. Better performance if training is performed over two runs of program (set EPOCHS to 20 and run twice) probably due to custom schedule.
3. 
Performance:
Models were given 24 inputs spaced over a 24hr period and predicted one timestep into the future. 
Baseline model (outputs most recent input as prediction) has a .089 average difference between target and predicted value. 
Transformer model has a average difference between target and predicted value.
