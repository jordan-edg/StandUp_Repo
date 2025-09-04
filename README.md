Hello! Welcome to the repository for the StandUp project.

To test the artefact, please make sure you run the entire repo folder in VSCode. StandUp.ipynb is where the retrieval script is. You can put your own query in test_query and give the model a try. The code used to create the query datasets has been left in the notebook, to show how I processed and saved them.

StandUp_Embeddings.parquet was the first attempt, vector_store_original was the first version of StandUp - and the one that the first query datasets were created with.
standup_embedding.py will be needed if you wish to create your own embeddings, or want to test the script. You will need your own OpenAI API key.

I have included all other datasets used in the project as a supplementary resource.

The TF-IDF classifier was an early feature, which no longer fit into what the project became. However, feel free to test it out and have a little play with it. 
