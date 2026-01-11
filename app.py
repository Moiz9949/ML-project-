import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Multiple Disease Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #111111;
    }
    .stButton > button {
        background-color: #111111;
        color: #ffffff;
        border: 1px solid #111111;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #2b2b2b;
        border-color: #2b2b2b;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: #111111;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #111111;
        border-bottom: 2px solid #111111;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #111111;
        color: #ffffff;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("Multiple Disease Data Explorer")
st.caption("Pick any diseases to explore. Use the tabs for a clean dashboard view.")

data_path = "Multiple Disease Data.csv"
df = load_data(data_path)

disease_candidates = [
    "HeartDisease",
    "Stroke",
    "Diabetic",
    "Asthma",
    "KidneyDisease",
]
disease_cols = [c for c in disease_candidates if c in df.columns]

if not disease_cols:
    st.error("No disease columns found in the CSV.")
    st.stop()

default_selection = disease_cols[:3]
selected = st.multiselect(
    "Select diseases (up to 5)",
    options=disease_cols,
    default=default_selection,
    max_selections=5,
)

if not selected:
    st.warning("Please select at least 1 disease.")
    st.stop()

def make_counts(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df_in[col]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("Status")
        .reset_index(name="Count")
    )

def render_pie(counts: pd.DataFrame, title: str) -> None:
    values = counts["Count"].tolist()
    labels = counts["Status"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)

def render_metrics(counts: pd.DataFrame) -> None:
    values = set(counts["Status"].astype(str))
    is_binary = values.issubset({"Yes", "No"})
    if not is_binary:
        top_label = counts.iloc[0]["Status"]
        top_count = int(counts.iloc[0]["Count"])
        total = int(counts["Count"].sum())
        pct = (top_count / total * 100) if total else 0.0
        c1, c2 = st.columns(2)
        c1.metric("Top Category", str(top_label))
        c2.metric("Top %", f"{pct:.1f}%")
        return

    yes_count = int(counts.loc[counts["Status"] == "Yes", "Count"].sum())
    no_count = int(counts.loc[counts["Status"] == "No", "Count"].sum())
    total = yes_count + no_count
    yes_pct = (yes_count / total * 100) if total else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Yes", f"{yes_count}")
    c2.metric("No", f"{no_count}")
    c3.metric("Yes %", f"{yes_pct:.1f}%")

tab_dashboard, tab_charts, tab_models = st.tabs(["Dashboard", "Charts", "Models"])

with tab_dashboard:
    for col in selected:
        st.subheader(col)
        counts = make_counts(df, col)
        render_metrics(counts)
        st.dataframe(counts, use_container_width=True, hide_index=True)

with tab_charts:
    for col in selected:
        st.subheader(col)
        counts = make_counts(df, col)
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(counts.set_index("Status"))
        with c2:
            render_pie(counts, f"{col} Distribution")


with tab_models:
    st.subheader("Model Results (Per Disease)")
    st.caption("Trains baseline models and reports test metrics for each selected disease.")


@st.cache_resource
def build_preprocessor(df_in: pd.DataFrame, target_col: str) -> ColumnTransformer:
    feature_df = df_in.drop(columns=[target_col])
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = feature_df.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


@st.cache_resource
def evaluate_models(df_in: pd.DataFrame, target_col: str) -> dict:
    X = df_in.drop(columns=[target_col])
    y = df_in[target_col].fillna("Missing")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(df_in, target_col)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=1),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    confusion = {}

    for name, model in models.items():
        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        average = "weighted"
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, average=average, zero_division=0),
                "Recall": recall_score(y_test, preds, average=average, zero_division=0),
                "F1": f1_score(y_test, preds, average=average, zero_division=0),
            }
        )
        confusion[name] = {
            "labels": sorted(y_test.unique().tolist()),
            "matrix": confusion_matrix(y_test, preds, labels=sorted(y_test.unique().tolist())),
        }

    return {
        "results": results,
        "confusion": confusion,
        "test_size": len(y_test),
    }


with tab_models:
    run_models = st.button("Run models for selected diseases")

    if run_models:
        for col in selected:
            st.markdown(f"### {col}")
            with st.spinner("Training and evaluating models..."):
                output = evaluate_models(df, col)

            results_df = pd.DataFrame(output["results"])
            results_df = results_df.sort_values("F1", ascending=False, ignore_index=True)

            st.write(f"Test size: {output['test_size']}")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                    "Precision": st.column_config.NumberColumn(format="%.4f"),
                    "Recall": st.column_config.NumberColumn(format="%.4f"),
                    "F1": st.column_config.NumberColumn(format="%.4f"),
                },
            )

            best_model = results_df.iloc[0]["Model"]
            st.write(f"Best model by F1: {best_model}")

            st.write("Confusion matrices:")
            for model_name, cm in output["confusion"].items():
                cm_df = pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])
                st.write(model_name)
                st.dataframe(cm_df, use_container_width=True)
