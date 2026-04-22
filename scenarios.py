import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


COLUMN = {
    "customer_id": "Kundenr",
    "shift_id": "Kjøreskift ID",
    "window_start": "Leveringsvindu fra",
    "window_end": "Leveringsvindu til",
    "service_start_actual": "Start Levering",
    "date": "Dato",
    "time_idx": "time_idx",
    "historical_volume_m3": "Levert volum (m3)",
}

PRED_COL = {
    "service_time_base_min": "predikert_leveringstid",
    "service_time_shock_min": "delivery_time_min_shock",
}


def add_bar_labels(axis, bars, fmt="{:.1f}", fontsize=7):
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize
        )


def minutes_since_midnight(ts):
    return int(ts.hour) * 60 + int(ts.minute)


def get_demo_dates_s2():
    return ["2025-06-18"]


def get_demo_dates_s3():
    return [
        "2023-06-05",
        "2024-05-06",
        "2025-11-03",
    ]


def load_scenario2_input():
    predictions_path = "../predictions.csv"
    history_path = "../merged_clean_7years.csv"

    pred = pd.read_csv(predictions_path, low_memory=False)
    pred.columns = pred.columns.str.strip()

    hist = pd.read_csv(history_path, low_memory=False)
    hist.columns = hist.columns.str.strip()

    pred[COLUMN["date"]] = pd.to_datetime(pred[COLUMN["date"]], errors="coerce")
    pred["merge_date"] = pred[COLUMN["date"]].dt.date

    hist[COLUMN["date"]] = pd.to_datetime(hist[COLUMN["date"]], errors="coerce")
    hist[COLUMN["window_start"]] = pd.to_datetime(hist[COLUMN["window_start"]], errors="coerce")
    hist[COLUMN["window_end"]] = pd.to_datetime(hist[COLUMN["window_end"]], errors="coerce")
    hist[COLUMN["service_start_actual"]] = pd.to_datetime(hist[COLUMN["service_start_actual"]], errors="coerce")
    hist["merge_date"] = hist[COLUMN["date"]].dt.date

    hist_for_merge = hist[
        [
            COLUMN["customer_id"],
            "merge_date",
            COLUMN["shift_id"],
            COLUMN["service_start_actual"],
            COLUMN["historical_volume_m3"],
            COLUMN["window_start"],
            COLUMN["window_end"],
        ]
    ].copy()

    scenario_input = (
        pred
        .merge(hist_for_merge, on=[COLUMN["customer_id"], "merge_date"], how="left")
        .dropna(subset=[COLUMN["shift_id"], COLUMN["service_start_actual"]])
        .rename(columns={COLUMN["historical_volume_m3"]: "hist_volume_m3"})
        .copy()
    )

    return scenario_input


def run_scenario_2(chosen_day, extra_pallets_largest, extra_pallets_second):
    PALLET_VOLUME_M3 = 1.2
    VEHICLE_CAPACITY_M3_BIL_HENGER = 42
    start_hour = 8
    end_hour = 16
    number_of_top_vehicles = 5

    scenario_input = load_scenario2_input()

    extra_volume_largest_stop_m3 = extra_pallets_largest * PALLET_VOLUME_M3
    extra_volume_second_stop_m3 = extra_pallets_second * PALLET_VOLUME_M3

    rows = scenario_input[scenario_input["merge_date"] == chosen_day].copy()
    rows["hist_volume_m3"] = pd.to_numeric(rows["hist_volume_m3"], errors="coerce").fillna(0)

    rows = rows.dropna(subset=[COLUMN["window_start"], COLUMN["window_end"]]).copy()
    rows = rows[rows[COLUMN["window_start"]].dt.hour.between(start_hour, end_hour)].copy()

    if len(rows) == 0:
        return {"summary": "No rows found for selected date.", "figure": None, "table": None}

    volume_by_vehicle = rows.groupby(COLUMN["shift_id"])["hist_volume_m3"].sum().sort_values(ascending=False)
    top_vehicle_ids = volume_by_vehicle.head(number_of_top_vehicles).index.tolist()

    rows = rows[rows[COLUMN["shift_id"]].isin(top_vehicle_ids)].copy()
    vehicle_order = top_vehicle_ids.copy()

    stops_volume = (
        rows.groupby(
            [
                COLUMN["shift_id"],
                COLUMN["customer_id"],
                COLUMN["time_idx"],
                COLUMN["window_start"],
                COLUMN["window_end"],
            ],
            as_index=False
        )
        .agg(base_volume_m3=("hist_volume_m3", "sum"))
    )

    stops_volume["rank_within_vehicle"] = (
        stops_volume.groupby(COLUMN["shift_id"])["base_volume_m3"].rank(method="first", ascending=False)
    )
    stops_volume["extra_volume_m3"] = 0.0
    stops_volume.loc[stops_volume["rank_within_vehicle"] == 1, "extra_volume_m3"] = extra_volume_largest_stop_m3
    stops_volume.loc[stops_volume["rank_within_vehicle"] == 2, "extra_volume_m3"] = extra_volume_second_stop_m3
    stops_volume["shock_volume_m3"] = stops_volume["base_volume_m3"] + stops_volume["extra_volume_m3"]

    vehicle_totals = (
        stops_volume.groupby(COLUMN["shift_id"], as_index=False)
        .agg(
            baseline_m3=("base_volume_m3", "sum"),
            scenario2_m3=("shock_volume_m3", "sum"),
        )
    )

    vehicle_totals["plot_order"] = vehicle_totals[COLUMN["shift_id"]].apply(lambda v: vehicle_order.index(v))
    vehicle_totals = vehicle_totals.sort_values("plot_order").reset_index(drop=True)

    x = np.arange(len(vehicle_totals))
    fig, axis = plt.subplots(figsize=(5, 2.8), dpi=120)

    bars_baseline = axis.bar(
        x - 0.2,
        vehicle_totals["baseline_m3"],
        0.4,
        label="Baseline",
    )

    bars_shock = axis.bar(
        x + 0.2,
        vehicle_totals["scenario2_m3"],
        0.4,
        label="Shock",
    )

    axis.axhline(VEHICLE_CAPACITY_M3_BIL_HENGER, linestyle="--", label="Capacity")
    axis.set_xticks(x)
    axis.set_xticklabels(vehicle_totals[COLUMN["shift_id"]].astype(str), fontsize=7)
    axis.set_ylabel("Volume (m3)", fontsize=8)
    axis.set_title("Scenario 2", fontsize=10)
    axis.legend(fontsize=7)

    add_bar_labels(axis, bars_baseline)
    add_bar_labels(axis, bars_shock)

    plt.tight_layout()

    vehicles_over_capacity = int((vehicle_totals["scenario2_m3"] > VEHICLE_CAPACITY_M3_BIL_HENGER).sum())

    summary = (
        f"Scenario 2 completed for {chosen_day}. "
        f"Vehicles analysed: {len(vehicle_totals)}. "
        f"Vehicles over capacity after shock: {vehicles_over_capacity}."
    )

    return {
        "summary": summary,
        "figure": fig,
        "table": vehicle_totals[[COLUMN["shift_id"], "baseline_m3", "scenario2_m3"]],
    }


def run_scenario_3(chosen_day, removed_vehicles, top_vehicles):
    kapasitet_m3 = 42
    fra_time = 8
    til_time = 15
    losningstid_sekunder = 10
    tillatt_venting_min = 600
    slutt_paa_dag_min = 24 * 60

    prediksjoner = pd.read_csv("../predictions.csv", low_memory=False)
    prediksjoner.columns = prediksjoner.columns.str.strip()

    historikk = pd.read_csv("../merged_clean_7years.csv", low_memory=False)
    historikk.columns = historikk.columns.str.strip()

    prediksjoner[COLUMN["date"]] = pd.to_datetime(prediksjoner[COLUMN["date"]], errors="coerce")
    historikk[COLUMN["date"]] = pd.to_datetime(historikk[COLUMN["date"]], errors="coerce")
    historikk[COLUMN["window_start"]] = pd.to_datetime(historikk[COLUMN["window_start"]], errors="coerce")
    historikk[COLUMN["window_end"]] = pd.to_datetime(historikk[COLUMN["window_end"]], errors="coerce")

    prediksjoner["merge_dato"] = prediksjoner[COLUMN["date"]].dt.date
    historikk["merge_dato"] = historikk[COLUMN["date"]].dt.date

    historikk_for_merge = historikk[
        [
            COLUMN["customer_id"],
            "merge_dato",
            COLUMN["shift_id"],
            COLUMN["historical_volume_m3"],
            COLUMN["window_start"],
            COLUMN["window_end"],
        ]
    ].copy()

    samlet_data = prediksjoner.merge(
        historikk_for_merge,
        on=[COLUMN["customer_id"], "merge_dato"],
        how="left"
    ).copy()

    samlet_data = samlet_data.rename(columns={COLUMN["historical_volume_m3"]: "volum_m3"})

    scenariodata = samlet_data[
        (samlet_data["merge_dato"] == chosen_day) &
        (samlet_data[COLUMN["window_start"]].dt.hour.between(fra_time, til_time))
    ].copy()

    scenariodata["volum_m3"] = pd.to_numeric(scenariodata["volum_m3"], errors="coerce").fillna(0.0)
    scenariodata["predikert_tid_min"] = pd.to_numeric(scenariodata["predikert_leveringstid"], errors="coerce")

    scenariodata = scenariodata.dropna(
        subset=["predikert_tid_min", COLUMN["shift_id"], COLUMN["customer_id"], COLUMN["window_start"], COLUMN["window_end"]]
    )

    if len(scenariodata) == 0:
        return {"summary": "No valid data found for selected date.", "figure": None, "table": None}

    topp_biler = (
        scenariodata.groupby(COLUMN["shift_id"])["volum_m3"]
        .sum()
        .sort_values(ascending=False)
        .head(top_vehicles)
        .index
        .tolist()
    )

    scenariodata = scenariodata[scenariodata[COLUMN["shift_id"]].isin(topp_biler)].copy()

    stopptabell = (
        scenariodata
        .groupby([COLUMN["customer_id"], COLUMN["window_start"], COLUMN["window_end"], COLUMN["shift_id"]], as_index=False)
        .agg(
            stopp_volum_m3=("volum_m3", "sum"),
            service_tid_min=("predikert_tid_min", "mean")
        )
    )

    stopptabell["vindu_start_min"] = stopptabell[COLUMN["window_start"]].apply(minutes_since_midnight).astype(int)
    stopptabell["vindu_slutt_min"] = stopptabell[COLUMN["window_end"]].apply(minutes_since_midnight).astype(int)

    før_tabell = (
        stopptabell
        .assign(**{COLUMN["shift_id"]: stopptabell[COLUMN["shift_id"]].astype(int)})
        .groupby(COLUMN["shift_id"], as_index=False)
        .agg(
            antall_stopp=(COLUMN["customer_id"], "size"),
            total_volum_m3=("stopp_volum_m3", "sum")
        )
        .sort_values("total_volum_m3", ascending=False)
    )

    fjernede_biler = (
        før_tabell
        .sort_values("total_volum_m3", ascending=True)
        .head(removed_vehicles)[COLUMN["shift_id"]]
        .astype(int)
        .tolist()
    )

    gjenværende_biler = [int(bil) for bil in topp_biler if int(bil) not in fjernede_biler]

    antall_biler = len(gjenværende_biler)
    antall_stopp = len(stopptabell)

    if antall_biler <= 0 or antall_stopp <= 0:
        return {"summary": "Invalid setup for Scenario 3.", "figure": None, "table": None}

    bil_til_indeks = {int(bil_id): i for i, bil_id in enumerate(gjenværende_biler)}
    indeks_til_bil = {i: bil_id for bil_id, i in bil_til_indeks.items()}

    etterspørsel_m3 = [0.0] + stopptabell["stopp_volum_m3"].tolist()
    service_tider = [0] + (
        pd.to_numeric(stopptabell["service_tid_min"], errors="coerce")
        .fillna(0)
        .round()
        .astype(int)
        .tolist()
    )

    vindu_start = [0] + stopptabell["vindu_start_min"].tolist()
    vindu_slutt = [slutt_paa_dag_min] + stopptabell["vindu_slutt_min"].tolist()

    reisetid = [
        [0 if i == j else 1 for j in range(antall_stopp + 1)]
        for i in range(antall_stopp + 1)
    ]

    manager = pywrapcp.RoutingIndexManager(antall_stopp + 1, antall_biler, 0)
    routing = pywrapcp.RoutingModel(manager)

    def reisetid_callback(fra_indeks, til_indeks):
        fra_node = manager.IndexToNode(fra_indeks)
        til_node = manager.IndexToNode(til_indeks)
        return int(reisetid[fra_node][til_node])

    reisetid_indeks = routing.RegisterTransitCallback(reisetid_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(reisetid_indeks)

    def etterspørsel_callback(fra_indeks):
        node = manager.IndexToNode(fra_indeks)
        return int(etterspørsel_m3[node] * 1000)

    etterspørsel_indeks = routing.RegisterUnaryTransitCallback(etterspørsel_callback)

    routing.AddDimensionWithVehicleCapacity(
        etterspørsel_indeks,
        0,
        [int(kapasitet_m3 * 1000)] * antall_biler,
        True,
        "Capacity"
    )

    def tid_callback(fra_indeks, til_indeks):
        fra_node = manager.IndexToNode(fra_indeks)
        til_node = manager.IndexToNode(til_indeks)
        return int(reisetid[fra_node][til_node] + service_tider[fra_node])

    tid_indeks = routing.RegisterTransitCallback(tid_callback)

    routing.AddDimension(
        tid_indeks,
        tillatt_venting_min,
        slutt_paa_dag_min,
        False,
        "Time"
    )

    tidsdimensjon = routing.GetDimensionOrDie("Time")

    for bil in range(antall_biler):
        tidsdimensjon.CumulVar(routing.Start(bil)).SetRange(0, slutt_paa_dag_min)
        tidsdimensjon.CumulVar(routing.End(bil)).SetRange(0, slutt_paa_dag_min)

    for node in range(1, antall_stopp + 1):
        indeks = manager.NodeToIndex(node)
        tidsdimensjon.CumulVar(indeks).SetRange(int(vindu_start[node]), int(vindu_slutt[node]))

    for node in range(1, antall_stopp + 1):
        opprinnelig_bil = int(stopptabell.loc[node - 1, COLUMN["shift_id"]])
        if opprinnelig_bil in bil_til_indeks:
            routing.VehicleVar(manager.NodeToIndex(node)).SetValue(bil_til_indeks[opprinnelig_bil])

    parametere = pywrapcp.DefaultRoutingSearchParameters()
    parametere.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parametere.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parametere.time_limit.FromSeconds(losningstid_sekunder)

    løsning = routing.SolveWithParameters(parametere)

    if løsning is None:
        return {"summary": "No feasible solution found in Scenario 3.", "figure": None, "table": None}

    stopp_til_bilindeks = {}

    for bil in range(antall_biler):
        indeks = routing.Start(bil)
        while not routing.IsEnd(indeks):
            node = manager.IndexToNode(indeks)
            if node != 0:
                stopp_til_bilindeks[node - 1] = bil
            indeks = løsning.Value(routing.NextVar(indeks))

    resultat = stopptabell.copy()
    resultat["før_bil"] = resultat[COLUMN["shift_id"]].astype(int)
    resultat["etter_bil"] = resultat.index.map(stopp_til_bilindeks).map(indeks_til_bil)

    flyttede_stopp = resultat[
        resultat["før_bil"].isin(fjernede_biler) & resultat["etter_bil"].notna()
    ].copy()

    rekkefølge = før_tabell[COLUMN["shift_id"]].astype(int).tolist()
    etiketter = rekkefølge[:]

    før_volum_map = før_tabell.set_index(før_tabell[COLUMN["shift_id"]].astype(int))["total_volum_m3"].to_dict()
    før_volum = np.array([float(før_volum_map.get(bil, 0.0)) for bil in etiketter])

    lagt_til_map = (
        flyttede_stopp.groupby("etter_bil")["stopp_volum_m3"].sum().to_dict()
        if len(flyttede_stopp) > 0 else {}
    )

    lagt_til_volum = np.array([float(lagt_til_map.get(bil, 0.0)) for bil in etiketter])

    fjernede_posisjoner = [etiketter.index(bil) for bil in fjernede_biler if bil in etiketter]

    grunn_volum_plot = før_volum.copy()
    lagt_til_plot = lagt_til_volum.copy()

    for posisjon in fjernede_posisjoner:
        grunn_volum_plot[posisjon] = 0.0
        lagt_til_plot[posisjon] = 0.0

    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    ax.bar(range(len(etiketter)), grunn_volum_plot, label="Original")
    ax.bar(range(len(etiketter)), lagt_til_plot, bottom=grunn_volum_plot, label="Reassigned")
    ax.axhline(kapasitet_m3, linestyle="--", label="Capacity")

    ax.set_xticks(range(len(etiketter)))
    ax.set_xticklabels([str(bil) for bil in etiketter], fontsize=7)
    ax.set_ylabel("Volume (m3)", fontsize=8)
    ax.set_title("Scenario 3", fontsize=10)
    ax.legend(fontsize=7)

    plt.tight_layout()

    summary = (
        f"Scenario 3 completed for {chosen_day}. "
        f"Removed vehicles: {fjernede_biler}. "
        f"Moved stops: {len(flyttede_stopp)}."
    )

    return {
        "summary": summary,
        "figure": fig,
        "table": flyttede_stopp[[
            COLUMN["customer_id"],
            "før_bil",
            "etter_bil",
            "stopp_volum_m3",
            "service_tid_min"
        ]].head(15),
    }