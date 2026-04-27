import streamlit as st
import pandas as pd

from scenarios import run_scenario_3, get_demo_dates_s3

st.set_page_config(page_title="Delivery Decision Support Tool", layout="centered")

st.title("Delivery Decision Support Tool")
st.write("Interactive prototype for scenario-based planning support.")

st.subheader("Scenario 3: Vehicle removal")

dates = get_demo_dates_s3()
chosen_day_text = st.selectbox("Choose date", dates)

removed_vehicles = st.number_input(
    "Number of vehicles to remove",
    value=2,
    step=1,
    min_value=1,
    max_value=5
)

top_vehicles = st.number_input(
    "Number of top vehicles to analyse",
    value=10,
    step=1,
    min_value=1,
    max_value=15
)

if st.button("Run Scenario 3"):
    try:
        chosen_day = pd.to_datetime(chosen_day_text).date()

        result = run_scenario_3(
            chosenDate=chosen_day,
            vehiclesToRemove=int(removed_vehicles),
            topVehicleCount=int(top_vehicles)
        )

        st.success("Done")
        st.write(result["summary"])

        if result["figure"] is not None:
            st.pyplot(result["figure"], clear_figure=True)

        if result["table"] is not None:
            st.dataframe(result["table"], use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")