import pandas as pd

def generate_suggestions(model, user_input):
    input_vector = [[
        user_input["age"],
        user_input["sleep"],
        user_input["study"],
        user_input["screen"],
        user_input["activity"],
        user_input["mood"],
        user_input["debt"],
        user_input["lifestyle"]
    ]]

    distances, indices = model.kneighbors(input_vector, n_neighbors=3)
    suggestion_db = pd.read_csv("data/suggestions.csv")
    return suggestion_db.iloc[indices[0]].to_dict(orient="records")
