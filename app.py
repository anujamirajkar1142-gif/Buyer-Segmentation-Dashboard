import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Parcl Co. Limited Buyer Segmentation Dashboard",
    page_icon="🏠",
    layout="wide"
)

df = pd.read_csv("Final_Buyer_Segmentation_Clean.csv")

st.title("Parcl Co. Limited Real Estate Buyer Segmentation Dashboard")

st.markdown("""
This dashboard provides insights into buyer behavior,
investment patterns and customer segmentation using
Machine Learning.
""")

st.sidebar.header("Filters")

segment = st.sidebar.multiselect(
    "Buyer Segment",
    options=df["Buyer_Segment"].unique(),
    default=df["Buyer_Segment"].unique()
)

country = st.sidebar.multiselect(
    "Country",
    options=df["country"].unique(),
    default=df["country"].unique()
)

client_type = st.sidebar.multiselect(
    "Client Type",
    options=df["client_type"].unique(),
    default=df["client_type"].unique()
)

filtered_df = df[
    (df["Buyer_Segment"].isin(segment)) &
    (df["country"].isin(country)) &
    (df["client_type"].isin(client_type))
]
st.subheader("Dashboard Overview")

col1,col2,col3 = st.columns(3)

col1.metric(
    "Total Buyers",
    len(filtered_df)
)

col2.metric(
    "Total Segments",
    filtered_df["Buyer_Segment"].nunique()
)

col3.metric(
    "Average Investment",
    f"${filtered_df['total_investment'].mean():,.0f}"
)

st.subheader("Buyer Segment Distribution")

fig1 = px.histogram(
    filtered_df,
    x="Buyer_Segment",
    color="Buyer_Segment"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

st.subheader("Investment Analysis by Segment")

fig2 = px.box(
    filtered_df,
    x="Buyer_Segment",
    y="total_investment",
    color="Buyer_Segment"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.subheader("Country-wise Buyer Distribution")

country_data = (
    filtered_df["country"]
    .value_counts()
    .reset_index()
)

country_data.columns = [
    "Country",
    "Count"
]

fig3 = px.bar(
    country_data,
    x="Country",
    y="Count"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.subheader("Customer Satisfaction Analysis")

fig4 = px.histogram(
    filtered_df,
    x="satisfaction_score",
    nbins=5
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

st.subheader("Age Distribution")

fig5 = px.histogram(
    filtered_df,
    x="Age",
    nbins=20
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

st.subheader("Segment Summary")

summary = (
    filtered_df
    .groupby("Buyer_Segment")
    [["Age",
      "satisfaction_score",
      "total_investment"]]
    .mean()
    .round(2)
)

st.dataframe(summary)

st.subheader("Dataset Preview")

st.dataframe(filtered_df.head(50))