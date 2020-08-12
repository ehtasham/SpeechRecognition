[//]: # (Image References)

[image1]: ./images/pipeline.png "ASR Pipeline"
[image2]: ./images/comparison.png "Model Comparison"
[image3]: ./images/spectrogram.png "Spectrogram"
[image4]: ./images/mfcc.png "MFCC"

# SpeechRecognition
End-to-End Automatic Speech Recognition(ASR) using Deep Neural Networks(DNN).

## Introduction  

In this notebook, we will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!  The completed pipeline will accept raw audio as input and return a predicted transcription of the spoken language.  The full pipeline is summarized in the figure below.

- **STEP 1** is a pre-processing step that converts raw audio to one of two feature representations that are commonly used for ASR. 
- **STEP 2** is an acoustic model which accepts audio features as input and returns a probability distribution over all potential transcriptions.
- **STEP 3** in the pipeline takes the output from the acoustic model and returns a predicted transcription.  



![ASR Pipeline][image1]



## Project Instructions
1. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
```
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf test-clean.tar.gz
mv flac_to_wav.sh LibriSpeech
cd LibriSpeech
chmod +x flac_to_wav.sh
./flac_to_wav.sh
```
2. Install the following packages
```
sudo apt-get install ffmpeg
wget http://launchpadlibrarian.net/348889634/libav-tools_3.4.1-1_all.deb
sudo dpkg -i libav-tools_3.4.1-1_all.deb

!pip install python_speech_features
!pip install SoundFile
```

3. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json

```
## The Data

We begin by investigating the dataset that will be used to train and evaluate the pipeline.  [LibriSpeech](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) is a large corpus of English-read speech, designed for training and evaluating models for ASR.  The dataset contains 1000 hours of speech derived from audiobooks.  We will work with a small subset in this project, since larger-scale data would take a long while to train.

## STEP 1: Acoustic Features for Speech Recognition

For this project, we won't use the raw audio waveform as input to the model.  Instead, we first performs a pre-processing step to convert the raw audio to a feature representation that has historically proven successful for ASR models.  The acoustic model will accept the feature representation as input.

In this project, you will explore two possible feature representations.

### Spectrograms

The first option for an audio feature representation is the [spectrogram](https://www.youtube.com/watch?v=_FatxGN3vAM). The implementation appears in the `utils.py` file in your repository.

The code returns the spectrogram as a 2D tensor, where the first (_vertical_) dimension indexes time, and the second (_horizontal_) dimension indexes frequency.  To speed the convergence of the algorithm, we have also normalized the spectrogram.  (we can see this quickly in the visualization below by noting that the mean value hovers around zero, and most entries in the tensor assume values close to zero.)

![Spectrogram][image3]

### Mel-Frequency Cepstral Coefficients (MFCCs)

The second option for an audio feature representation is [MFCCs](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).  Just as with the spectrogram features, the MFCCs are normalized in the supplied code.

The main idea behind MFCC features is the same as spectrogram features: at each time window, the MFCC feature yields a feature vector that characterizes the sound within the window.  Note that the MFCC feature is much lower-dimensional than the spectrogram feature, which could help an acoustic model to avoid overfitting to the training dataset. 

![MFCC][image4]

## STEP 2: Neural Network model Architectures and comparison

#### Model 0: A single GRU layer.
This is just a simple model with a single layer of rnn, so it doesn't converge and since both its training and validation loss are high and close to each other so the model is underfitting and is unable to learn. The reason this is happenning is because it is a very simple model with only 16,617 parameters.

#### Model 1: RNN + TimeDistributed Dense
In this model we added TimeDistributed Dense layer which significantly improved the number of parameters to 223,829 so the model was able to learn from the data which is why this model is performing much better than the simple model. However the difference between the training loss and validation loss is big which suggests that the model is overfitting.


#### Model 2: CNN + RNN + TimeDistributed Dense
Adding a CNN layer improved the model's performance on training data and it's loss is lowest so far on training data, however on validation data it started to perfrom well but half way throug the validation loss started to increase which means that the model is overfitting.

#### Model 3: Deeper RNN + TimeDistributed Dense
This is a significantly bigger model so it performs better however the CNN layer is missing which is why it is overfitting, It started off pretty well since both the training and validation loss were close and were decreasing but then later on validation loss started to lag behind which overfit the model.

#### Model 4: Bidirectional RNN + TimeDistributed Dense
Using bidirectional model performs better than simple RNN but it's performance is bad as compared to model_2 which uses CNN. Also this model is overfitting.

![Model Comparison][image2]


#### Final Model
##### Architecture

- 3 bidirectional RNN layers with GRU and recurrent_dropout=0.1
- BatchNormalization after each bidirectional RNN layer
- Time Distributed Dense layer at the end.

##### Intuition Behind Final Model
I started building my final model by using the insights from the previous models, as CNN and Bidirectional model was performing better so I combined a CNN layer with 3 bidirectional layer and BatchNormalization after each layer, but the results were surprising as this model was not converging and the loss started to increase after a few epochs. 

I changed my model after this by removing the CNN layer and using 3 bidirectional layers with BatchNormalization and By using both dropouts 
(dropout for inputs and dropout for recurrent state) in the recurrent layers as suggested in the notebook, but this model was overfitting alot, so I removed the dropout for inputs and kept the dropout for the recurrent state to reduce overfitting. Initially I was using LSTM but the with LSTM none of the models were converging so I used GRU instead of LSTM. Also at the start I was not using the MFCC features but I experimented by using the MFCC feature which resulted in an improved performance.

This model performed much better but it is still overfitting.

## STEP 3: Predictions

#### Training set
##### True transcription: 
he was in a fevered state of mind owing to the blight his wife's action threatened to cast upon his entire future

##### Predicted transcription:
he was in ha fevered stat ofmind owing to the blight his whitee sactiond prene tocast ipon hes indtire fetur


#### Training set
##### True transcription
so it is said anders

##### Predicted transcription:
sowadeads sat ander

