# imdb_551

A bunch of different models on 25k Large Movie Review (IMDb) dataset.
Results are reported in the attached paper. 

Install dependencies: pip install -r requirements.txt

IF INITIAL RUN, in retrieve_and_pre function in each model file, make fromsave=False, as this will initialize the preprocessing step in the feature extraction pipeline, as well as save a local copy of the preprocessed data onto the machine. For the biLSTM model, set retrieve_and_pre parameter tfidfpca=False, as we want the function to return raw preprocessed sentences instead of scaled TFxIDF features. 

To run model: python3 -m model.<insert-module-name>

