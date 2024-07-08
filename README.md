The Python code, developed on Google Colab, is designed to create an emotion recognition system using machine learning. First, it imports essential libraries such as librosa, soundfile, os, glob, pickle, numpy, and specific functions from sklearn (train_test_split from sklearn.model_selection, MLPClassifier from sklearn.neural_network, and accuracy_score from sklearn.metrics). The function extract_features is defined to extract key audio features like Chroma, Mel-frequency, and MFCC, which are crucial for determining emotions in speech. Emotions are mapped to specific codes based on the RAVDESS dataset, focusing on the observed emotions: happy, calm, angry, and sad. The load_data function is responsible for loading and splitting the dataset into training and testing sets. It iterates through the audio files, extracts relevant features using extract_feature, and only includes files with observed emotions. The dataset is then split into training and testing sets with a specified test_size of 50%. The MLPClassifier is initialized with specific parameters: alpha set to 0.015, batch_size to 256, epsilon to 1e-08, hidden_layer_sizes to (150,), learning_rate to 'adaptive', and max_iter to 550, to effectively classify the emotions. Finally, a graphical user interface (GUI) is constructed to make the project a practical application.

Here is a snapshot of the complete code:

python
Copy code
# Import necessary libraries
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Mapping emotions to specific codes
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# List of observed emotions
observed_emotions = ['happy', 'calm', 'angry', 'sad']

# Function to load data and extract features
def load_data(test_size=0.25):
    x, y = [], []
    for file in glob.glob("/content/drive/MyDrive/ravdess/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = load_data(test_size=0.5)

# Initialize the MLPClassifier with specified parameters
model = MLPClassifier(
    alpha=0.015,
    batch_size=256,
    epsilon=1e-08,
    hidden_layer_sizes=(150,),
    learning_rate='adaptive',
    max_iter=550
)
This concise explanation includes all the major steps and details of the code, making it easy to understand the workflow of the emotion recognition system.

لا حط كلام وبعدين كود 
ChatGPT
The provided Python code, developed on Google Colab, is designed to create an emotion recognition system using machine learning. The process begins by importing essential libraries such as librosa, soundfile, os, glob, pickle, numpy, and specific functions from sklearn (train_test_split from sklearn.model_selection, MLPClassifier from sklearn.neural_network, and accuracy_score from sklearn.metrics).

python
Copy code
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
Next, emotions are mapped to specific codes based on the RAVDESS dataset, focusing on the observed emotions: happy, calm, angry, and sad.

python
Copy code
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['happy', 'calm', 'angry', 'sad']
The function extract_features is defined to extract key audio features like Chroma, Mel-frequency, and MFCC, which are crucial for determining emotions in speech.

The load_data function is responsible for loading and splitting the dataset into training and testing sets. It iterates through the audio files, extracts relevant features using extract_feature, and only includes files with observed emotions.

python
Copy code
def load_data(test_size=0.25):
    x, y = [], []
    for file in glob.glob("/content/drive/MyDrive/ravdess/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
The dataset is then split into training and testing sets with a specified test_size of 50%.

python
Copy code
x_train, x_test, y_train, y_test = load_data(test_size=0.5)
The MLPClassifier is initialized with specific parameters: alpha set to 0.015, batch_size to 256, epsilon to 1e-08, hidden_layer_sizes to (150,), learning_rate to 'adaptive', and max_iter to 550, to effectively classify the emotions.

python
Copy code
model = MLPClassifier(
    alpha=0.015,
    batch_size=256,
    epsilon=1e-08,
    hidden_layer_sizes=(150,),
    learning_rate='adaptive',
    max_iter=550
)
Finally, a graphical user interface (GUI) is constructed to make the project a practical application, providing a user-friendly interface as illustrated in Figure 10. This concise explanation includes all the major steps and details of the code, making it easy to understand the workflow of the emotion recognition system.
