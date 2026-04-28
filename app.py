import streamlit as st
import pandas as pd
from scenarios import run_scenario_3, get_demo_dates_s3

st.title("Delivery Decision Support Tool")
date = st.selectbox("Date", get_demo_dates_s3())
remove = st.number_input("Remove", 1, 5, 2)
top = st.number_input("Top", 1, 15, 10)

if st.button("Run"):
    try:
        r = run_scenario_3(pd.to_datetime(date).date(), remove, top)
        st.success("Done")
        st.write(r["summary"])
        st.pyplot(r["figure"])
        st.dataframe(r["table"])
    except:
        st.error("No solution")
