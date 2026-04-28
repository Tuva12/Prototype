import streamlit as st
import pandas as pd
from scenarios import run_scenario_3, get_demo_dates_s3

date = st.selectbox("Date", get_demo_dates_s3())
remove = st.number_input("Remove vehicles", 1, 5, 2)
top = st.number_input("Top vehicles", 1, 15, 10)

if st.button("Run"):
    result = run_scenario_3(
        pd.to_datetime(date).date(),
        int(remove),
        int(top)
    )

    st.write(result["summary"])
    st.pyplot(result["figure"])
    st.dataframe(result["table"])
