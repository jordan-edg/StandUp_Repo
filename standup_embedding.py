import os
import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI
from dotenv import load_dotenv


df = pd.read_csv("StandUp_Dataset_Final.csv")

# Using API Key to initialise OpenAI client

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




def embed_text(text: str) -> list[float]:
    """This function generates an embedding function for a given text string."""
    resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return resp.data[0].embedding

# Looping through each row of the dataset to generate embeddings; including a loading bar so that the process can be monitored.
embeddings = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows"):
    vec = embed_text(row["content"])
    embeddings.append(vec)


# Constructing a dataframe from the embeddings and all the relevant metadata for StandUp.

emb_df = pd.DataFrame(embeddings)
emb_df["id"]      = df["id"].values
emb_df["title"]   = df["title"].values
emb_df["section"] = df["section"].values
emb_df["heading"] = df["heading"].values
emb_df["content"] = df["content"].values
emb_df["url"] = df["url"].values
emb_df["category"] = df["category"].values

# Saving to a parquet for efficient storage and retrieval.

emb_df.to_parquet("StandUp_Embeddings_Test.parquet", index=False)

print("Done! All embeddings generated and saved.")

