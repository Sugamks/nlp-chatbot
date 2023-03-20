# nlp-chatbot
A chatbot which uses NLP to reply to the questions.
There are 2 models in this chatbot
  1. Question Answering module
  2. Knowledge Base

# Question Answering Module
This module uses FeedForward Neural Network to train the questions and select the answers from intents.json file.
  1. First create a .json file and save the questions related to your work.
  2. Download nltk.punkt 
      ```
      The current Python version does not download torch which is required. So I used Google's Colab to run my code.
      If you are using Google's Colab Then you should import 2 packages.
      !pip install pyngrok==4.1.1
      !pip install flask_ngrok
      ```
      ``` python
      pip install nltk
      python # type this in your command line interface
      >>> import nltk
      >>> nltk.download('punkt')
      ```
      ```
      Output:
            [nltk_data] Downloading package punkt to /root/nltk_data...
            [nltk_data]   Unzipping tokenizers/punkt.zip.
            True
      ```
   3. Run train.py
      ```
      Output: x patterns
              n tags: [-,-,-,-,-,-,-,-,-]
              y unique stemmed words: ["","","","","",""]
              y n
              Epoch [100/1000], Loss: 0.6628
              Epoch [200/1000], Loss: 0.5638
              Epoch [300/1000], Loss: 0.0785
              Epoch [400/1000], Loss: 0.0803
              Epoch [500/1000], Loss: 0.0063
              Epoch [600/1000], Loss: 0.0025
              Epoch [700/1000], Loss: 0.0008
              Epoch [800/1000], Loss: 0.0039
              Epoch [900/1000], Loss: 0.0012
              Epoch [1000/1000], Loss: 0.2527
              final loss: 0.2527
              ```
   4. Run kb.py 
      ```
      Output: [nltk_data] Downloading package punkt to /root/nltk_data...
      [nltk_data]   Package punkt is already up-to-date!
      [nltk_data] Downloading package wordnet to /root/nltk_data...
      [nltk_data] Downloading package omw-1.4 to /root/nltk_data...
      [nltk_data] Downloading package stopwords to /root/nltk_data...
      [nltk_data]   Unzipping corpora/stopwords.zip.
      ```
   5. Run app.py
      ```
      After importing NGROK 
      Use this link https://dashboard.ngrok.com/login to download your NGROK interface.
      From the interface you will get your authtoken id
      ```
      ```
        !ngrok authtoken #your authtoken ID
      ```
      ```
      Use the above code before running app.py
      You will get a flask API link through which you can access the chatbot.
