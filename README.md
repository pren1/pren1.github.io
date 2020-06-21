Introduction

When the vtubers are streaming together, their voices sometimes get mixed. In this condition, it could be hard for the fansub members to figure out what the target vtuber is saying. So, we would like to propose a model that could split the voices come from different vtubers. In this way, the heavy burden of the fansub could get reliefed. Moreover, once the voice split process is done, people could run the multiple people voice activity detection immediatley, which is a long-lasting demand.

Thus, in this project, we come up with a model that could split the mixed two vtuber voices rather efficiently. More vtubers will be taken into consideration in the future work. Besides, we need more people to contribute to this project before we can really make it. Please feel free to contact me if you are willing to get your time wasted on these things :D

Related work

The main idea of the model comes from this Google paper. In this paper, they are able to split the target person's voice using the D-vector. The pytorch code of this paper exists at here. However, we found that the ir model does not really work for the Japanese vtubers. That is, the dataset they used is not suitable for our task. So, it becomes necessary for us to build the dataset from scrach and modify the model to pursue better performance.

Process pipline

In this section, I will try to go over the whole complex piplline. However, for more details, please refer to the code mentioned in the following subsections.

1. Data collection
   The code of this part could be found at here.
   1. Data selection
      So, we would like to split the mixed voices from speaker A and B. To do this, we first need to obtain the audio that only contains A's voice and B's voice. Then, as claimed in the google's paper, one could easiliy mix the two people's voices and  build a synthesis dataset to train the model. Thus, at the very beginning, we need to perform the audio selection. That is, we go to the youtube channels of the two speaks, and find the vedio that meet the requirement above.
   2. Data download
      The youtube-dl is utilized for this process. We directly extract the opus format audio from the video using the --extract-audio command provided by the youtube-dl.
   3. Raw audio signal preprocess
      Since the video may contains back ground music, one should definitely remove the bgm. Fortunetaly, the Spleeter model is ready to use, and it works well. The audios are then splitted into 5-minutes slices. After that, we downsamples the audio signals from 48000Hz to 8000Hz.
2. Build dataset
   The code of this part could be found at here.
   1. Clip data
      5-mintunes audio is still too much for the limited ram we have. Thus, we clip the data into 3-second slices this time.
   2. Data cleaning
      If a speaker does not speak more than 1.5 seconds within an audio slices, we just remove that audio slice. As it turns out, this data cleaning process is quite important for the subsequent model performance.
   3. Data mixture and data augmentation
      For better peroformance, we perform the data augmentation here to make sure the model works better. That is, for each audio wave, we first normlaize the signals:
          s1_target /= np.max(np.abs(s1_target))
          s2 /= np.max(np.abs(s2))
      Then, we multiply the two wave with two different ratios that are sampled from a uniform distribution. After that, the two signals are added up, and they are normlaized again:
          mixed = w1 + w2
          norm = np.max(np.abs(mixed)) * 1.1
          w1, w2, mixed = w1/norm, w2/norm, mixed/norm
      Additionally, we use the short time furior transformation technique to transfer the audio singals to the frequency domain. After this process, what we get is actually a bunch of 2-d images, which would be processed by the Convolutional neural network.
3. Model structure
   The code of this part could be found at here. Here is the model structure. For the input to the model, note that we also need to specify the target speaker that needs to be splitted. In this condition, an  embedding vector that specifies the speaker is utilized as an extra input. For more details of this model, please refer to the original paper and our modified code.
   
   Note that we modify the original model structure by doing the following things.
   1. Another bidirectional LSTM is added to enhance the model ability
   2. Multiple Dropout layer is added to avoid overfit
   3. The attention mechinism is implemented so that the model could focus on different parts of the CNN output when it generates the mask. (If you do not understand what is the mask, please read goolge's paper first!)


