import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# Added only selected dates for demo.
def get_demo_dates_s3():
    return [
        "2025-04-22",
        "2025-06-11",
        "2025-12-16",
        "2025-04-25",
        "2025-06-13"]



def run_scenario_3(chosenDate, vehiclesToRemove, topVehicleCount):
    # Copy paste the scenario 3 code here, that is identical to the scenario 3 notebook.
    
    fromHour = 8
    toHour = 15
    capacity = 42
    endOfDay = 24 * 60
    solveTime = 10
    
    customerColumn = "Kundenr"
    dateColumn = "Dato"
    shiftColumn = "Kjøreskift ID"
    startWindow = "Leveringsvindu fra"
    endWindow = "Leveringsvindu til"
    predColumn = "predikert_leveringstid"
    volumeColumn = "Levert volum (m3)"
    
    def toMinutes(timeValue):
        return timeValue.hour * 60 + timeValue.minute
    
    def addLabels(axis, bars, values, bottom=None):
        for number, bar in enumerate(bars):
            value = values[number]
    
            if value <= 0:
                continue
    
            if bottom is None:
                height = value
            else:
                height = bottom[number] + value
    
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.3,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9)
    
    predData = pd.read_csv("predictions.csv", low_memory=False)
    historyData = pd.read_csv("merged_clean_7years.csv", low_memory=False)
    
    predData.columns = predData.columns.str.strip()
    historyData.columns = historyData.columns.str.strip()
    
    predData[dateColumn] = pd.to_datetime(predData[dateColumn], errors="coerce")
    historyData[dateColumn] = pd.to_datetime(historyData[dateColumn], errors="coerce")
    historyData[startWindow] = pd.to_datetime(historyData[startWindow], errors="coerce")
    historyData[endWindow] = pd.to_datetime(historyData[endWindow], errors="coerce")
    
    predData["mergeDate"] = predData[dateColumn].dt.date
    historyData["mergeDate"] = historyData[dateColumn].dt.date
    
    historySmall = historyData[[
        customerColumn,
        "mergeDate",
        shiftColumn,
        volumeColumn,
        startWindow,
        endWindow
    ]].copy()
    
    data = predData.merge(historySmall, on=[customerColumn, "mergeDate"], how="left")
    data = data.rename(columns={volumeColumn: "volume"})
    chosenDate = pd.to_datetime(chosenDate).date()
    
    scenarioData = data[
        (data["mergeDate"] == chosenDate) &
        (data[startWindow].dt.hour.between(fromHour, toHour))].copy()
    
    scenarioData["volume"] = pd.to_numeric(
        scenarioData["volume"], errors="coerce").fillna(0)
    
    scenarioData["serviceTime"] = pd.to_numeric(
        scenarioData[predColumn], errors="coerce")
    
    scenarioData = scenarioData.dropna(
        subset=[
            customerColumn,
            shiftColumn,
            startWindow,
            endWindow,
            "serviceTime"])
    print("Rows:", len(scenarioData))
    
    topVehicles = (
        scenarioData.groupby(shiftColumn)["volume"]
        .sum()
        .sort_values(ascending=False)
        .head(topVehicleCount)
        .index
        .tolist())
    
    scenarioData = scenarioData[
        scenarioData[shiftColumn].isin(topVehicles)].copy()
    
    stops = (
        scenarioData
        .groupby(
            [customerColumn, startWindow, endWindow, shiftColumn],
            as_index=False)
        .agg(
            stopVolume=("volume", "sum"),
            serviceTime=("serviceTime", "mean")))
    
    stops["startMin"] = stops[startWindow].apply(toMinutes)
    stops["endMin"] = stops[endWindow].apply(toMinutes)
    
    vehicleTable = (
        stops.groupby(shiftColumn, as_index=False)
        .agg(totalVolume=("stopVolume", "sum"))
        .sort_values("totalVolume", ascending=False))
    
    removedVehicles = (
        vehicleTable.sort_values("totalVolume")
        .head(vehiclesToRemove)[shiftColumn]
        .astype(int)
        .tolist())
    
    remainingVehicles = [
        int(vehicle)
        for vehicle in topVehicles
        if int(vehicle) not in removedVehicles]
    
    print("Removed vehicles:", removedVehicles)
    print("Remaining vehicles:", remainingVehicles)
    
    vehicleToNumber = {
        vehicle: number
        for number, vehicle in enumerate(remainingVehicles)}
    
    numberToVehicle = {
        number: vehicle
        for vehicle, number in vehicleToNumber.items()}
    
    stopCount = len(stops)
    vehicleCount = len(remainingVehicles)
    
    volumeList = [0] + stops["stopVolume"].tolist()
    serviceList = [0] + stops["serviceTime"].fillna(0).round().astype(int).tolist()
    startList = [0] + stops["startMin"].tolist()
    endList = [endOfDay] + stops["endMin"].tolist()
    
    travelTime = [
        [0 if startStop == endStop else 1 for endStop in range(stopCount + 1)]
        for startStop in range(stopCount + 1)]
    
    manager = pywrapcp.RoutingIndexManager(
        stopCount + 1,
        vehicleCount,
        0)
    
    routing = pywrapcp.RoutingModel(manager)
    
    def travelCallback(startIndex, endIndex):
        startStop = manager.IndexToNode(startIndex)
        endStop = manager.IndexToNode(endIndex)
    
        return travelTime[startStop][endStop]
    
    def volumeCallback(startIndex):
        stopNumber = manager.IndexToNode(startIndex)
    
        return int(volumeList[stopNumber] * 1000)
    
    def timeCallback(startIndex, endIndex):
        startStop = manager.IndexToNode(startIndex)
        endStop = manager.IndexToNode(endIndex)
    
        return travelTime[startStop][endStop] + serviceList[startStop]
    
    travelIndex = routing.RegisterTransitCallback(travelCallback)
    routing.SetArcCostEvaluatorOfAllVehicles(travelIndex)
    
    volumeIndex = routing.RegisterUnaryTransitCallback(volumeCallback)
    
    routing.AddDimensionWithVehicleCapacity(
        volumeIndex,
        0,
        [int(capacity * 1000)] * vehicleCount,
        True,
        "Capacity")
    
    timeIndex = routing.RegisterTransitCallback(timeCallback)
    
    routing.AddDimension(
        timeIndex,
        600,
        endOfDay,
        False,
        "Time")
    
    timeDim = routing.GetDimensionOrDie("Time")
    
    for stopNumber in range(1, stopCount + 1):
        stopIndex = manager.NodeToIndex(stopNumber)
    
        timeDim.CumulVar(stopIndex).SetRange(
            int(startList[stopNumber]),
            int(endList[stopNumber]))
    
    for stopNumber in range(1, stopCount + 1):
        oldVehicle = int(stops.loc[stopNumber - 1, shiftColumn])
    
        if oldVehicle in vehicleToNumber:
            routing.VehicleVar(
                manager.NodeToIndex(stopNumber)
            ).SetValue(vehicleToNumber[oldVehicle])
    
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(solveTime)
    solution = routing.SolveWithParameters(params)
    
    if solution is None:
        raise RuntimeError("No feasible solution found")
    print("Solution found")

    
    stopToVehicle = {}
    for vehicleNumber in range(vehicleCount):
        routeIndex = routing.Start(vehicleNumber)
    
        while not routing.IsEnd(routeIndex):
            stopNumber = manager.IndexToNode(routeIndex)
            if stopNumber != 0:
                stopToVehicle[stopNumber - 1] = vehicleNumber
            routeIndex = solution.Value(routing.NextVar(routeIndex))
    result = stops.copy()
    result["beforeVehicle"] = result[shiftColumn].astype(int)
    result["afterVehicle"] = result.index.map(stopToVehicle).map(numberToVehicle)


    
    movedStops = result[
        result["beforeVehicle"].isin(removedVehicles) &
        result["afterVehicle"].notna()].copy()
    print("Moved stops:", len(movedStops))
    
    labels = vehicleTable[shiftColumn].astype(int).tolist()
    volumeBefore = vehicleTable["totalVolume"].to_numpy().copy()
    addedVolume = np.zeros(len(labels))
    
    for rowNumber, row in movedStops.iterrows():
        vehicle = row["afterVehicle"]
        if vehicle in labels:
            addedVolume[labels.index(vehicle)] += row["stopVolume"]
    removedPositions = []
    
    for vehicle in removedVehicles:
        if vehicle in labels:
            position = labels.index(vehicle)
            removedPositions.append(position)
            volumeBefore[position] = 0
            addedVolume[position] = 0
    originalVolume = vehicleTable["totalVolume"].to_numpy()
    
    xValues = np.arange(len(labels)) * 1.25
    barWidth = 0.8
    fig, axis = plt.subplots(figsize=(11, 4))
    
    barsBefore = axis.bar(
        xValues,
        volumeBefore,
        width=barWidth,
        label="Original load")
    
    barsAdded = axis.bar(
        xValues,
        addedVolume,
        width=barWidth,
        bottom=volumeBefore,
        label="Reallocated load")
    
    axis.axhline(
        24,
        color="orange",
        linestyle="--",
        label="Truck only capacity")
    
    axis.axhline(
        42,
        color="red",
        linestyle="--",
        label="Truck + trailer capacity")
    
    for position in removedPositions:
        axis.bar(
            xValues[position],
            originalVolume[position],
            width=barWidth,
            color="none",
            edgecolor="black",
            linestyle="--",
            linewidth=2,
            label="Removed vehicle"
            if position == removedPositions[0]
            else None)
    
    addLabels(axis, barsBefore, volumeBefore)
    addLabels(axis, barsAdded, addedVolume, bottom=volumeBefore)
    
    axis.set_xticks(xValues)
    axis.set_xticklabels(
        [str(vehicle) for vehicle in labels],
        rotation=20)
    
    axis.set_ylabel("Total volume (m3)")
    axis.set_title(f"Scenario 3 - {chosenDate}")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.show()
    
    table = movedStops[[
        customerColumn,
        startWindow,
        endWindow,
        "beforeVehicle",
        "afterVehicle",
        "stopVolume",
        "serviceTime"]].head(30).copy()
    
    table[startWindow] = pd.to_datetime(
        table[startWindow],
        errors="coerce").dt.strftime("%H:%M")
    
    table[endWindow] = pd.to_datetime(
        table[endWindow],
        errors="coerce").dt.strftime("%H:%M")
    
    table["stopVolume"] = pd.to_numeric(
        table["stopVolume"],
        errors="coerce").round(1)
    
    table["serviceTime"] = pd.to_numeric(
        table["serviceTime"],
        errors="coerce").round(1)
    
    table = table.rename(columns={
        customerColumn: "Customer",
        startWindow: "Window start",
        endWindow: "Window end",
        "beforeVehicle": "Original vehicle",
        "afterVehicle": "New vehicle",
        "stopVolume": "Stop volume",
        "serviceTime": "Service time"})
    
    print(table.to_string(index=False))


    
    # return the results
    return {
        "summary":
            f"Scenario 3 completed for {chosenDate}. "
            f"Removed vehicles: {removedVehicles}. "
            f"Moved stops: {len(movedStops)}.",
        "figure": fig,
        "table": table}