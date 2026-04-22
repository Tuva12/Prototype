import streamlit as st
import pandas as pd

from scenarios import run_scenario_2, run_scenario_3, get_demo_dates_s2, get_demo_dates_s3

st.set_page_config(page_title="Delivery Decision Support Tool", layout="centered")

st.title("Delivery Decision Support Tool")
st.write("Interactive prototype for scenario-based planning support.")

scenario = st.selectbox(
    "Choose scenario",
    ["Scenario 2 - Volume shock", "Scenario 3 - Vehicle removal"]
)

if scenario == "Scenario 2 - Volume shock":
    st.subheader("Scenario 2: Volume shock")

    dates = get_demo_dates_s2()
    chosen_day_text = st.selectbox("Choose date", dates)

    extra_1 = st.number_input("Extra pallets at largest stop", value=5, step=1, min_value=0, max_value=10)
    extra_2 = st.number_input("Extra pallets at second-largest stop", value=4, step=1, min_value=0, max_value=10)

    if st.button("Run Scenario 2"):
        try:
            chosen_day = pd.to_datetime(chosen_day_text).date()

            result = run_scenario_2(
                chosen_day=chosen_day,
                extra_pallets_largest=int(extra_1),
                extra_pallets_second=int(extra_2),
            )

            st.success("Done")
            st.write(result["summary"])

            if result["figure"] is not None:
                st.pyplot(result["figure"], clear_figure=True)

            if result["table"] is not None:
                st.dataframe(result["table"], use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")


if scenario == "Scenario 3 - Vehicle removal":
    st.subheader("Scenario 3: Vehicle removal")

    dates = get_demo_dates_s3()
    chosen_day_text = st.selectbox("Choose date", dates)

    removed_vehicles = st.number_input(
        "Number of vehicles to remove",
        value=2,
        step=1,
        min_value=1,
        max_value=3
    )

    top_vehicles = st.number_input(
        "Number of top vehicles to analyse",
        value=8,
        step=1,
        min_value=5,
        max_value=10
    )

    if st.button("Run Scenario 3"):
        try:
            chosen_day = pd.to_datetime(chosen_day_text).date()

            result = run_scenario_3(
                chosen_day=chosen_day,
                removed_vehicles=int(removed_vehicles),
                top_vehicles=int(top_vehicles),
            )

            st.success("Done")
            st.write(result["summary"])

            if result["figure"] is not None:
                st.pyplot(result["figure"], clear_figure=True)

            if result["table"] is not None:
                st.dataframe(result["table"], use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")