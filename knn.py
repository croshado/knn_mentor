# Step 1: Import libraries
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Step 2: Create the mentor dataset
mentors = pd.DataFrame([
    {"mentor_id": 1, "name": "Siddhi", "subjects": "CR,LR", "style": "Visual", "college": "NLSIU", "prep_level": "Advanced"},
    {"mentor_id": 2, "name": "Akshat Shraf", "subjects": "LR,English", "style": "Auditory", "college": "NALSAR", "prep_level": "Intermediate"},
    {"mentor_id": 3, "name": "Shreyans H Singh", "subjects": "CR,Math", "style": "Kinesthetic", "college": "NLU Delhi", "prep_level": "Advanced"},
    {"mentor_id": 4, "name": "Vinayak Kedia", "subjects": "CR,LR,GK", "style": "Visual", "college": "NLU Jodhpur", "prep_level": "Intermediate"},
    {"mentor_id": 5, "name": "Tyush Agarwal", "subjects": "English,Math", "style": "Auditory", "college": "NLSIU", "prep_level": "Advanced"},
    {"mentor_id": 6, "name": "Sara Joshi", "subjects": "LR,GK", "style": "Visual", "college": "NLU Delhi", "prep_level": "Intermediate"},
    {"mentor_id": 7, "name": "Anirban Dutta", "subjects": "CR,Math", "style": "Kinesthetic", "college": "NLSIU", "prep_level": "Advanced"},
    {"mentor_id": 8, "name": "Akanksha", "subjects": "CR,LR,English", "style": "Auditory", "college": "NALSAR", "prep_level": "Beginner"},
    {"mentor_id": 9, "name": "Sara Srivastava", "subjects": "LR,GK", "style": "Visual", "college": "NLU Jodhpur", "prep_level": "Intermediate"},
    {"mentor_id": 10, "name": "Rahul Gulati", "subjects": "CR,English,GK", "style": "Kinesthetic", "college": "NLU Delhi", "prep_level": "Intermediate"},
    {"mentor_id": 11, "name": "Sreyaans Shukla", "subjects": "LR,Math", "style": "Visual", "college": "NLSIU", "prep_level": "Advanced"},
    {"mentor_id": 12, "name": "Tejas", "subjects": "CR,LR", "style": "Auditory", "college": "NLU Jodhpur", "prep_level": "Beginner"},
    {"mentor_id": 13, "name": "Aayush", "subjects": "Math,English", "style": "Kinesthetic", "college": "NALSAR", "prep_level": "Intermediate"},
    {"mentor_id": 14, "name": "Siya", "subjects": "GK,LR", "style": "Visual", "college": "NLSIU", "prep_level": "Beginner"},
    {"mentor_id": 15, "name": "Laaliya", "subjects": "CR,Math", "style": "Kinesthetic", "college": "NLU Delhi", "prep_level": "Intermediate"},
    {"mentor_id": 16, "name": "Dhruv Nair", "subjects": "LR,GK,English", "style": "Auditory", "college": "NLU Jodhpur", "prep_level": "Advanced"},
    {"mentor_id": 17, "name": "Ishika", "subjects": "Math,English", "style": "Visual", "college": "NLU Delhi", "prep_level": "Intermediate"},
    {"mentor_id": 18, "name": "Vishnesh Shekhar", "subjects": "CR,LR,GK", "style": "Kinesthetic", "college": "NLSIU", "prep_level": "Advanced"}
])

# Step 3: Preprocessing function
def preprocess(df, is_mentor=True):
    df = df.copy()
    df["CR"] = df["subjects"].apply(lambda x: int("CR" in x))
    df["LR"] = df["subjects"].apply(lambda x: int("LR" in x))
    df["Math"] = df["subjects"].apply(lambda x: int("Math" in x))
    df["English"] = df["subjects"].apply(lambda x: int("English" in x))
    df["GK"] = df["subjects"].apply(lambda x: int("GK" in x))
    
    df = pd.get_dummies(df, columns=["college", "style", "prep_level"])
    
    if is_mentor:
        df = df.drop(columns=["mentor_id", "name", "subjects"])
    else:
        df = df.drop(columns=["subjects"])
    
    return df

# Step 4: Preprocess mentor dataset
mentor_features = preprocess(mentors)

# Step 5: Input from the aspirant
def get_aspirant_input():
    print("Please enter the aspirant's details:")
    subjects = input("Preferred Subjects (e.g., CR,LR,GK): ").strip()
    college = input("Target College (e.g., NLSIU, NALSAR, NLU Delhi): ").strip()
    style = input("Learning Style (Visual, Auditory, Kinesthetic): ").strip()
    prep_level = input("Preparation Level (Beginner, Intermediate, Advanced): ").strip()
    
    aspirant = {
        "subjects": subjects,
        "college": college,
        "style": style,
        "prep_level": prep_level
    }
    return aspirant

# Step 6: Get the aspirant input and preprocess it
aspirant_input = get_aspirant_input()
aspirant_df = pd.DataFrame([aspirant_input])
aspirant_vector = preprocess(aspirant_df, is_mentor=False)
aspirant_vector = aspirant_vector.reindex(columns=mentor_features.columns, fill_value=0)

# Step 7: Apply KNN with customized feature weights
def weighted_knn(aspirant, mentors, feature_weights_dict):
    weights = pd.Series(feature_weights_dict)
    weights = weights.reindex(mentors.columns, fill_value=1)
    weighted_mentors = mentors * weights
    weighted_aspirant = aspirant * weights

    model = NearestNeighbors(n_neighbors=3, metric='cosine')
    model.fit(weighted_mentors)
    distances, indices = model.kneighbors(weighted_aspirant)
    return indices

# Step 8: Define feature weights (adjust based on importance)
feature_weights = {
    "CR": 3, "LR": 3, "Math": 3, "English": 3, "GK": 3,  # Subjects
    "college_NLSIU": 3, "college_NALSAR": 3, "college_NLU Delhi": 3, "college_NLU Jodhpur": 3,  # College
    "style_Auditory": 1.5, "style_Visual": 1.5, "style_Kinesthetic": 1.5,  # Learning style
    "prep_level_Beginner": 2, "prep_level_Intermediate": 2, "prep_level_Advanced": 2  # Prep level
}

# Step 9: Get weighted KNN recommendations
weighted_indices = weighted_knn(aspirant_vector, mentor_features, feature_weights)

# Step 10: Show recommended mentors
print("\n🎯 Top 3 Recommended Mentors:")
for idx in weighted_indices[0]:
    row = mentors.iloc[idx]
    print(f"- {row['name']} from {row['college']} (Preferred Subjects: {row['subjects']}, Style: {row['style']}, Prep Level: {row['prep_level']})")
