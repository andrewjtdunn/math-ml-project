import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def top_x_categories(df_column, x):
    """
    Create X new columns based on top X frequent categories (e.g. top 25 dog breeds)
    """
    # Get counts of each unique value in the column
    value_counts = df_column.value_counts()

    # Extract top X values
    top_x_values = value_counts.head(x).index.tolist()

    # Create new column
    new_column = df_column.apply(lambda val: val if val in top_x_values else "Other")

    return new_column


def convert_to_months(age_str):
    """
    Convert age column (str) in austin shelter dataset to standard unit (months)
    in float. If age is less than 1 month, it is set to 0
    """
    # Check if value is already a float
    if isinstance(age_str, float):
        return max(0, age_str)  # Set negative values to 0

    # Extract numerical values and units
    value, unit = age_str.split()

    # Convert to months
    if unit.startswith("week"):
        return max(0, float(value) / 4.33)  # Approximate number of weeks in a month
    elif unit.startswith("month"):
        return max(0, float(value))
    elif unit.startswith("day"):
        return max(0, float(value) / 30)  # Approximate number of days in a month
    elif unit.startswith("year"):
        return max(0, float(value) * 12)  # Approximate number of months in a year
    else:
        return None


def one_hot_encode(df, column_name):
    """
    Returns dataframe that has one hot encoded columns such as top X categories
    """

    dummies = pd.get_dummies(df[column_name], prefix=None)
    dummies = dummies.astype(int)

    # Concatenate dummy variables with the original dataframe
    df = pd.concat([df, dummies], axis=1)

    return df


def load_shelter_data():
    """
    Currently cleans and returns the subset of data for cats
    """

    df1 = pd.read_csv("../data/Austin_Animal_Center_Intakes.csv")
    df1 = df1.drop(columns=["Found Location", "Name"])

    df2 = pd.read_csv("../data/Austin_Animal_Center_Outcomes.csv")
    df2 = df2.drop(
        columns=[
            "Name",
            "DateTime",
            "Date of Birth",
            "MonthYear",
            "Animal Type",
            "Sex upon Outcome",
            "Age upon Outcome",
            "Breed",
            "Color",
        ]
    )

    intake_outcome_df = pd.merge(df1, df2, on="Animal ID", how="left")
    intake_outcome_df = intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Animal Type"] == "Dog")
        | (intake_outcome_df.loc[:, "Animal Type"] == "Cat")
    ]

    intake_outcome_df["Outcome Labels"] = 6
    intake_outcome_df["Stray"] = 0
    intake_outcome_df["Owner Surrender"] = 0
    intake_outcome_df["Public Assist"] = 0
    intake_outcome_df["Abandoned"] = 0
    intake_outcome_df["Euthanasia Requested"] = 0
    intake_outcome_df["Female"] = 0
    intake_outcome_df["Intact"] = 0
    intake_outcome_df["Cond Normal"] = 0
    intake_outcome_df["Cond Aged"] = 0
    intake_outcome_df["Cond Med"] = 0  # Med Attn, Med Urgen, Medical, Injured
    # Neurologic, Sick, Panleuk, Agonal
    intake_outcome_df["Cond Behavior"] = 0
    intake_outcome_df["Cond Other"] = 0  # Space, Unknown, Other
    intake_outcome_df["Cond Feral"] = 0
    intake_outcome_df["Cond Neonatal"] = 0
    intake_outcome_df["Cond Nursing"] = 0
    intake_outcome_df["Cond Pregnant"] = 0

    intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Outcome Type"] == "Euthanasia")
        | (intake_outcome_df.loc[:, "Outcome Type"] == "Died"),
        "Outcome Labels",
    ] = 0
    intake_outcome_df.loc[
        (
            (intake_outcome_df.loc[:, "Outcome Type"] == "Adoption")
            & (
                (intake_outcome_df.loc[:, "Outcome Subtype"] != "Foster")
                | (intake_outcome_df.loc[:, "Outcome Subtype"].isna())
            )
        ),
        "Outcome Labels",
    ] = 1
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Outcome Subtype"] == "Foster", "Outcome Labels"
    ] = 2
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Outcome Type"] == "Return to Owner", "Outcome Labels"
    ] = 3
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Outcome Type"] == "Transfer", "Outcome Labels"
    ] = 4
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Outcome Type"].isna(), "Outcome Labels"
    ] = 5  # non-outcomes

    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Intake Type"] == "Stray", "Stray"
    ] = 1
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Intake Type"] == "Owner Surrender", "Owner Surrender"
    ] = 1
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Intake Type"] == "Public Assist", "Public Assist"
    ] = 1
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Intake Type"] == "Abandoned", "Abandoned"
    ] = 1
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Intake Type"] == "Euthanasia Requested",
        "Euthanasia Requested",
    ] = 1
    intake_outcome_df["Female"] = np.where(
        intake_outcome_df["Sex upon Intake"].str.contains("Female"), 1, 0
    )
    intake_outcome_df["Intact"] = np.where(
        intake_outcome_df["Sex upon Intake"].str.contains("Intact"), 1, 0
    )
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Intake Condition"] == "Normal", "Cond Normal"
    ] = 1
    intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Intake Condition"] == "Aged", "Cond Aged"
    ] = 1
    intake_outcome_df.loc[
        (
            (intake_outcome_df.loc[:, "Intake Condition"] == "Med Attn")
            | (intake_outcome_df.loc[:, "Intake Condition"] == "Med Urgent")
            | (intake_outcome_df.loc[:, "Intake Condition"] == "Medical")
            | (intake_outcome_df.loc[:, "Intake Condition"] == "Injured")
            | (intake_outcome_df.loc[:, "Intake Condition"] == "Neurologic")
            | (intake_outcome_df.loc[:, "Intake Condition"] == "Sick")
            | (intake_outcome_df.loc[:, "Intake Condition"] == "Panleuk")
            | (intake_outcome_df.loc[:, "Intake Condition"] == "Agonal")
        ),
        "Cond Med",
    ] = 1

    intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Intake Condition"] == "Behavior"), "Cond Behavior"
    ] = 1
    intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Intake Condition"] == "Space"), "Cond Other"
    ] = 1
    intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Intake Condition"] == "Unknown"), "Cond Other"
    ] = 1
    intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Intake Condition"] == "Other"), "Cond Other"
    ] = 1
    intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Intake Condition"] == "Feral"), "Cond Feral"
    ] = 1
    intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Intake Condition"] == "Neonatal"), "Cond Neonatal"
    ] = 1
    intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Intake Condition"] == "Nursing"), "Cond Nursing"
    ] = 1
    intake_outcome_df.loc[
        (intake_outcome_df.loc[:, "Intake Condition"] == "Pregnant"), "Cond Pregnant"
    ] = 1

    # clean colors (600 -> 430)
    intake_outcome_df["Color"] = intake_outcome_df["Color"].apply(
        lambda x: "/".join(sorted(x.split("/")))
    )  # standardize duplicates e.g. brown/white and white/brown -> brown/white
    intake_outcome_df["Color"] = intake_outcome_df["Color"].apply(
        lambda x: x.split("/")[0]
        if len(set(x.split("/"))) == 1
        else "/".join(sorted(set(x.split("/"))))
    )

    # clean breeds (2600+ -> 2000)
    intake_outcome_df["Breed"] = intake_outcome_df["Breed"].apply(
        lambda x: "/".join(sorted(x.split("/")))
    )  # standardize duplicates

    # convert age
    intake_outcome_df["Age in Months"] = intake_outcome_df["Age upon Intake"].apply(
        convert_to_months
    )

    df_dog = intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Animal Type"] == "Dog"
    ].copy()
    df_cat = intake_outcome_df.loc[
        intake_outcome_df.loc[:, "Animal Type"] == "Cat"
    ].copy()

    df_cat["TopBreeds"] = top_x_categories(df_cat["Breed"], 15)

    df_dog["TopBreeds"] = top_x_categories(df_dog["Breed"], 50)

    # one hot encode breeds
    df_dog = one_hot_encode(df_dog, "TopBreeds")
    df_cat = one_hot_encode(df_cat, "TopBreeds")

    df_cat_numeric = df_cat.iloc[:, 12:].copy()
    df_cat_numeric = df_cat_numeric.drop(columns="TopBreeds")
    df_cat_numeric = df_cat_numeric[
        df_cat_numeric["Outcome Labels"] != 6
    ]  # drop outcomes we don't care about

    return df_cat_numeric.to_numpy()


# Oversampling did not yield a more predictive model, so we excluded this method
def over_sample_data(X, y, seed=123):
    """ """
    ros = RandomOverSampler(random_state=seed)
    train_X_resampled, train_y_resampled = ros.fit_resample(X, y)

    return train_X_resampled, train_y_resampled
