import pandas as pd
# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch

def get_embeddings(text, model):
    with torch.no_grad():
        text = text.to(model.device)
        return model(**text)
    # if len(model.tokenizer(text)["input_ids"]) >= 384:
    #     pass
    # return model.encode(text)

def wait_for_debugger(port: int = 5678):
    """
    Pauses the program until a remote debugger is attached. Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(
        f"Waiting for client to attach on port {port}... NOTE: if using docker, you need to forward the port with -p {port}:{port}."
    )
    debugpy.wait_for_client()

def main():
    #wait_for_debugger()
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    dataset = pd.read_csv("imdb.csv")
    dataset["embeddings"] = None
    count = 0
    for i in tqdm(range(0, 100), desc=f"writing embeddings"):
        tokenized = tokenizer(dataset.iloc[i]["text"], padding=True, return_tensors="pt")
        if len(tokenized["input_ids"][0]) >= 512:
            count += 1
        #dataset.at[i, "embeddings"] = get_embeddings(tokenized, model)

    print(count)



    # batch_size = 16
    # for i in tqdm(range(0, 100, batch_size), desc=f"writing embeddings"):
    #     batch = dataset.iloc[i:i+batch_size]
    #     tokenized = tokenizer(batch["text"].tolist(), padding=True, truncation=True, return_tensors="pt")
    #     embeddings = get_embeddings(tokenized, model)
    #     for i, embedding in enumerate(embeddings):
    #         dataset.at[i, "embeddings"] = embedding




    # wait_for_debugger()
    # model = SentenceTransformer("all-mpnet-base-v2")
    # print(model.max_seq_length)
    # dataset = pd.read_csv("imdb.csv")
    # dataset["embeddings"] = None
    # for i in tqdm(range(100), desc=f"writing embeddings"):
    #     embeddings = get_embeddings(dataset.iloc[i]["text"], model)
    #     dataset.at[i, "embeddings"] = embeddings

    # # Filter rows with empty embeddings
    # dataset = dataset[dataset["embeddings"].notnull()]
    # dataset.to_csv("imdb_embeddings.csv")

if __name__ == "__main__":
    main()
