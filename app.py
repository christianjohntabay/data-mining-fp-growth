import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

st.title("FP-Growth Data Mining Project")

st.header("1. Upload Transaction Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Uploaded Data")
    st.dataframe(df)

    # Convert items column from string → list
    st.header("2. Cleaning Data")
    df['items'] = df['items'].apply(lambda x: [i.strip() for i in x.split(',')])
    st.write("Converted 'items' into list format:")
    st.dataframe(df.head())

    # Encode data for FP-Growth
    st.header("3. Transaction Encoding")
    te = TransactionEncoder()
    te_array = te.fit(df['items']).transform(df['items'])
    encoded_df = pd.DataFrame(te_array, columns=te.columns_)
    st.write("Encoded dataset:")
    st.dataframe(encoded_df.head())

    # FP-Growth
    st.header("4. FP-Growth Frequent Itemsets")
    minsup = st.slider("Select minimum support", 0.01, 1.0, 0.05)

    frequent_itemsets = fpgrowth(encoded_df, min_support=minsup, use_colnames=True)
    st.write("Generated frequent itemsets:")
    st.dataframe(frequent_itemsets)

    st.success("FP-Growth completed!")
