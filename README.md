# Newspaper Classifier
# Setup
1. Make sure to install all required dependencies in advance:
    'pip install -r requirements.txt'
2. Navigate to the main directory and run the setup programme to download necessary files.
    * Enter 'python3 run demo.py' to download the NLTK data (it's a lot). 
    * When prompted, type 'd' to download.
    * Then, type 'all' to download all packages.
    * And finally, 'q' to quit.
3. After this, navigate to the Demo directory and enter 'streamlit run demo.py' to open the app.

# Intro
This application demonstrates a range of classification models to determine whether a specified newspaper headline falls into one of the following 4 categories:
    
    * Business
    * Science and Technology
    * Entertainment
    * Health

The available models were trained on a BBC News dataset, which can be found [here](https://www.kaggle.com/c/learn-ai-bbc/data).

The available models are:

    * Logistic Regression
    * Naive Bayes
    * One Vs Rest
    * Simple Feedforward Neural Network
    * ALBERT (Lightweight Transformers model)

Each of these models were trained on a preprocessed version of the above dataset and achieved validation accuracies of between 65% to 92% with little to no fine-tuning.
Hence, next steps for this project would be to tweak these models to improve their accuracy where possible and potentially assess other models also. 
In particular, XGBoost may be worth consideration for this task.

Specific areas that could be fine-tuned include:
# Pre-Processing
The pre-processing script used on the given dataset was simple but effective, however there's space for further improvement and experimentation with lemmatisation, stop words, feature engineering, etc.

# Neural Network Architecture
For both, the simple neural network and the ALBERT model, a naive, base architecture was used to gain quick results. in the case of the simple neural network this was enough to achieve a validation accuracy of nearly 92%, however the ALBERT classifier only achieved 65% accuracy.
Testing new architectures for both of these models would likely improve the accuracy for both these models. In particular, the simple NN has far too many parameters (48 million). This should be drastically reduced for a more efficent model - Its current size is overkill.

Also, I used the recommended learning rate, optimisers, batch size, etc from the TensorFlow documentation, however there may also be scope here to tweak these variables to improve the accuracy of the models.

# Additional Info
An early version of this demo application is available at https://text-classification-1de1de243d7d.herokuapp.com/. 
This version only includes ML models; Logistic Regression, Naive Bayes, & One vs Rest, and it skips the preprocessing stage for input text due to issues with downloading nltk data within the production environemnt in Heroku (this would be a bug to fix in future deployed demos).
