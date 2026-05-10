# Visual Storytelling with Cross-Modal Attention and Bi-directional  GRU

In this project there is done improvement in a multimodal grounded sequence prediction of baseline on the `daniel3303/StoryReasoning` dataset. This model recives four story frames and descriptions for each , then prediction is done for the fifth image and description for it.

# Quick Links
- **[Experiments Notebook](experiment_notebook.ipynb)**
- **[Baseline Results/Outcomes](Results/baseline/)**
- **[Experiment 1 Results/Outcomes](Results/Experiment_1)**
- **[Experiment 2 Results/Outcomes](Results/Experiment_2/)**
- **[Experiment 3 Results/Outcomes](Results/Experiment_3/)**


# Innovation Summary 

# Experiment-1 
I have done changes in the Grounding/fusion part of the baseline architecture. The orginal baseline model have got the simple concatentation where the text latent features and visual latent features are merged together.This procedure blends image and text information, although it doesn't allow the model to clearly understand the relationship within particular words and equivalent visual region. In this experiment Cross-Modal Attention is introduced to enhance the visual-text grounding.The spatial visual feature mapping through the visual encoder function as values and keys, whereas the contextual text token attributes from the LSTM are used as queries.This technique helps the model to concentrate on essential areas of the image, while generating the grounded text visualization. The main aim of this modification is to improve visual-text grounding, lower the referent hallucination as well as to help model to produce text which is better aligned to visual content of the narrative frames.In the outcomes, BLEU-4 score of Cross-Modal Attention is lower than baseline but the final training loss is a slightly higher than the baseline. 

# Experiment-2 
I have done changes in the Sequence Predictor component of the baseline architecture. The baseline model used unidirectional GRU to analyze the merged visual-text data.This indicates that the model is only reading the story sequence only in forward direction, from the initial input frame to the final input frame. In this experiment2, I replaced the unidirectional GRU with Bidirectional GRU. It allows the model to process the sequence in both backward and forward directions which read the 4 input frames. Final forward and backward As a outocme, information can be used from the initial phase till end phase of the input sequence by the model while making the final description of the story. The main aim of this modification is to enhance temporal perception and story coherence before forecasting the next frame and text. 

# Experiment- 3 
I have done this experiment to compare BiGRU with the BiLSTM and do the testing whether the LSTM cell state gives superior temporal modeling for story sequence.Baseline Model Sequence predictor was modified, unidirectional GRU is used in baseline after combining the visiual and text latent features. I exchanged the unidirectional GRU with bidirectional LSTM. The BiLSTM retrieves the fused visual-text sequence and analyzes the 4 input frames both backward and forward.In this process the last forward and backward hidden sequence are joined, combined together to attention context vector, as well as then anticipated back within the latent space for text and image prediction. 


# Key Results 
| Metric         |    Baseline  |   Experiment- 1    |   Experiment- 2         |    Experiment- 3      |
--------------------------------------------------------------------------------------------------------
| Training Loss  |    3.9996    |   3.9976           |        4.3588           |    3.9957             |
| BLEU-4         |    0.0244    |   0.0154           |        0.0133           |    0.0000             |     
       

![Experiment- 1 ( Cross-Modal Attention Overlay )](results/Experiment_1/attention_overlay.png)
 

# Findings
There are some findings related to the experiments, initial when all experiments where done compared to the baseline on 10 epochs, experiment 1 used Cross-Modal Attention, while Experiment 2 Used a Bidirectional GRU. The inital output showed that Experiment 1 and Experiment 2 have got slightly improved BLUE-4 compared to baseline but the loss was not less than the baseline.

While doing investigation, I found that the pretrained text autoencoder was fully frozen. Due to freezed decoder limited the ability of the model for adaption to the multimodal problem when the sequence predictor produces the intended description utilizing the text decoder.Therefore, I kept the text encoder frozen however allowed the text decoder to train. This adjustement decreased the model's entire training loss. The BLUE-4 and final trainning loss of baseline was accurate after that and the experiment 1 and experiment 2 have got changes. Experiment 1 got slightly improved BLUE-4 than baseline and final trainning loss was slightly higher  while on other hand experiment 2 have both BLUE-4 and final training loss worse than baseline.

After that to get more in depth why the experiments were not giving the better loss the training of 15 epochs were done for baseling , Experiment 1 and Experiment2. After that both of the experiment 1 and 2 got the slightly lower final training loss than the baseline.Experiment 1 reduced the loss from 3.9996 to 3.9976, and Experiment 2 reduced the loss to 3.9957.This indicates that both Cross-Modal Attention and BiGRU had the capacity to optimize the  training goal moderately superior than the baseline with additional training time.

Neverthless, BLUE-4 reduced for both experiments. The baseline have got the BLUE-4 score of 0.0244, while Experiment 1 got 0.0154 and Experiment 2 got 0.0133. This occured while BLUE-4 is determined from the end of the produced sentence, although the training loss is determined with teacher forcing, where the model obtains the proper prior words.Hence, the model can decrease token-level loss however still produce wording that departs from the orginal text.This clearly shows that the lower training loss will not always  necessarily lead into superior text output. 
 
Similarly, Experiment 3- BiLSTM was added to do comparision with  Experiment 2- BiGRU. When the model was trained on 10 epochs Experiment 3 gained a better BLUE-4 score than Experiment 2, while its final training loss was a bit higher. But after training 15 epochs , BiLSTM achieved a same loss to BiGRU, yet its BLUE-4 dropped to 0.0000. It indicates that although extended training increased teacher-forced reduction, it have no impact on published text during BLUE-4 analysis.

Overall, the outcome demonstrate that BLEu-4 does not always get improved by reducing training loss.This is becasue training loss is intended utilizing teacher forcing, whereas BLEU-4 analyzes the final initiated sentence as well as involves exact phrase matches. 

# How to Reproduce

1. `pip install - r requirements.txt`
2. Open `experiment_notebook.ipynb`
3. First run all the cells sequentially for the Baseline as Codes are comment out and the do it for Experiment 1. (Takes around 30 mins to baseline and 30 Mins for Experiment 1 on GPU)
