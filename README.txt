Delivery Decision Support Tool - Tuva Stokka Pettersen


-- This is an interactive prototype
-- The tool shows how machine learning and optimisation can support delivery planning through different scenarios

It combines:

* Predicted service times from a TFT model
* Optimisation with Google OR-Tools
* Interactive interface with Streamlit


The prototype was made to test scenario-based planning support.

This demo focuses on Scenario 3:

* Remove vehicles from the fleet
* Reallocate stops to remaining vehicles
* Show updated vehicle loads


-----Features-----

* Choose date
* Choose number of vehicles to remove
* Choose number of vehicles to analyse
* View load per vehicle
* Compare capacity limits
* View moved stops in a table


-----Input Files-----

* predictions.csv
Contains predicted service times from the TFT model

* demo_data.csv
Contains historical logistics data for selected dates:
  * customer IDs
  * vehicle IDs
  * delivery volume
  * delivery time windows


-----Limitations of prototype-----

This prototype is a simplified proof-of-concept.

* Only Scenario 3 is included in the app version
* Scenario 1 and Scenario 2 can be added later, as any other future scenarios needed.
* Only a small number of demo dates are available, for demo purpose only.
* Real travel distances are not used
* Travel time is simplified to 1 minute between all stops
* Fake / simplified coordinates are not used in this version
* The prototype is made for demonstration purposes, not daily operations.


-------How to Run-------

Install packages:
* pip install streamlit pandas numpy matplotlib ortools

Run app:
streamlit run app.py


-----Project Files-----

* app.py = user interface
* scenarios.py = scenario Logic (reused code from scenario analysis in the thesis)
* predictions.csv = model predictions
* demo_data.csv = spesific selected dates, of historical data

-----Thesis Code-----

* The full bachelor thesis and analysis code can be found here:
  https://github.com/Tuva12/ScenarioAnalysisInGroceryLogistics


