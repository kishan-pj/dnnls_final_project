# Visual Storytelling with Cross-Modal Attention and Bi-directional  GRU

In this project there is done improvement in a multimodal grounded sequence prediction of baseline on the `daniel3303/StoryReasoning` dataset. This model recives four story frames and descriptions for each , then prediction is done for the fifth image and description for it.

# Quick Links
- **[Experiments Notebook](experiment_notebook.ipynb)**
- **[Baseline Results/Outcomes](Results/baseline/)**
- **[Experiment 1 Results/Outcomes](Results/Experiment_1)**

# Innovation Summary 
Experiment-1 - I have done changes in the Grounding Module of the Baseline Architecture. The orginal baseline model have got the simple concatentation where the text latent features and visual latent features are merged together.This procedure blends image and text information, although it doesn't allow the model to clearly understand the relationship within particular words and equivalent visual region. In this experiment the basic fusion approach is swapped with Cross-Modal Attention. The setup is done for the text token embedding where it serve as queries, during the structural visual feature maps operate as keys and values. This technique helps the model to concentrate on essential areas of the image, while generating the grounded text visualization. The main aim of this modification is to improve visual-text grounding, lower the referent hallucination as well as to help model to produce text which is better aligned to visual content of the narrative frames. 


# Key Results 
| Metric         |    Baseline  |   Experiment- 1    |           Change                 |
-----------------------------------------------------------------------------------------
| Training Loss  |    4.3596    |   4.3628           |     +0.032 (Slightly high)       |
| BLEU-4         |    0.0000    |   0.0125           |     +0.0125 (Improved)           |            
 


# InFinding
The innovation have of cross-modal attention enhanced the model's capability to line up visual and textual information. Compared to the  baseline model which got the BLEU-4 Score of 0.0000, On the other hand Experiment 1 obtained a score of 0.0125, marking the beginning of better text genertation. Moreover, the attention mechanism permitted the model to concentrate on applicable visual area, indicating improved grounding among attributes and generated text. While, the final training loss was slightly enhanced , the progress in BLUE-4 verifies that the quality of text predictions has been improved. 


# How to Reproduce

1. `pip install - r requirements.txt`
2. Open `experiment_notebook.ipynb`
3. First run all the cells sequentially for the Baseline as Codes are comment out and the do it for Experiment 1. (Takes around 30 mins to baseline and 30 Mins for Experiment 1 on GPU)
