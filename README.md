The provided Python code, developed on Google Colab, involves several steps to create an emotion recognition system using machine learning. Firstly, necessary libraries such as librosa, soundfile, os, glob, pickle, numpy, and specific functions from sklearn (train_test_split from sklearn.model_selection, MLPClassifier from sklearn.neural_network, and accuracy_score from sklearn.metrics) were imported. The function extract_features was defined to extract audio features like Chroma, Mel-frequency, and MFCC, which are crucial for identifying emotions in speech. The code then listed several emotions (neutral, calm, happy, sad, angry, fearful, disgust, and surprised) and assigned specific values to each emotion based on the RAVDESS dataset. However, only the observed emotions (happy, sad, calm, and angry) were considered for the analysis.

The function load_data was defined to collect these features from the audio files, with a test size of 20%. The dataset was split into training (75%) and testing (25%) sets using the MLPClassifier for classification. The model then predicted the emotions for the test set (y_pred), and the accuracy of the system was calculated and printed using the accuracy_score function. Finally, a graphical user interface (GUI) was developed to make the project a real application, providing a user-friendly interface for interaction.






