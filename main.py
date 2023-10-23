from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Tuple, Dict, List
import json

# https://geopy.readthedocs.io/en/stable/#module-geopy.distance
from geopy.distance import distance
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

GEOLOCATOR = Nominatim(user_agent="supercharger_locator")
# https://www.tesla.com/model3
TESLA_RANGE = 535  # kilometers

SUPERCHARGER_RAW_CSV_DATA = Path(__file__).parent / "data/tesla_superchargers.csv"
SUPERCHARGER_LOC_DATA = Path(__file__).parent / "data/tesla_superchargers.json"

START_STATION = "Palo Alto, CA - Stanford Shopping Center Supercharger"
END_STATION = "Boston Supercharger"


def check_lat_long(lat_long: Tuple[float, float]):
    # Latitude-Longitude values based on map of US
    return (
        lat_long[0] > 20
        and lat_long[0] < 50
        and lat_long[1] < -60
        and lat_long[1] > -130
    )


@dataclass
class SuperchargerData:
    charging_station_name: str
    city: str
    state: str
    zipcode: int
    street_address: str

    @property
    def lat_long(self) -> Tuple[float, float]:
        # From https://geopy.readthedocs.io/en/stable/#module-geopy.geocoders
        location = GEOLOCATOR.geocode(self.street_address)
        return (location.latitude, location.longitude)

    @property
    def valid_lat_long(self) -> bool:
        return check_lat_long(self.lat_long)

    @property
    def as_dict(self) -> Dict:
        dict = self.__dict__
        dict.update({"lat_long": self.lat_long})
        return dict


def query_positions():
    json_data = {"charger_loc_data": []}

    data_headers = None

    with open(SUPERCHARGER_RAW_CSV_DATA, "r") as file:
        csvreader = csv.reader(file, delimiter=",")
        for row in csvreader:
            # Get name of each column from CSV (first row)
            if not data_headers:
                data_headers = row
                continue

            charger_data = SuperchargerData(
                **{data_headers[i]: row[i] for i in range(len(row))}
            )

            # Catching 2 Different errors here:
            #   1. Attribute errors for when GeoPy fails to find a location for the inputted address
            #   2. GeocoderUnavailable for when GeoPy times out when searching for the inputted location
            try:
                # This check is needed becaused geopy sometimes returns coordinates that are
                # outside the bounds of the US
                if charger_data.valid_lat_long:
                    json_data["charger_loc_data"].append(charger_data.as_dict)
            except (AttributeError, GeocoderUnavailable):
                continue

    with open(SUPERCHARGER_LOC_DATA, "w") as file:
        json.dump(json_data, file, indent=4)


def compute_distances():
    with open(SUPERCHARGER_LOC_DATA, "r") as file:
        json_data = json.load(file)

    json_data["dist_pairs"] = []
    json_data["dist_to_goal"] = {}

    for loc in json_data["charger_loc_data"]:
        if loc["charging_station_name"] == END_STATION:
            dest_lat_long = loc["lat_long"]
            break
    else:
        raise ValueError(f"Could not find Lat-Long data for {END_STATION}")

    for i in range(len(json_data["charger_loc_data"])):
        station_1 = json_data["charger_loc_data"][i]
        for j in range(i + 1, len(json_data["charger_loc_data"])):
            station_2 = json_data["charger_loc_data"][j]

            d = int(distance(station_1["lat_long"], station_2["lat_long"]).km)
            if d < TESLA_RANGE:
                json_data["dist_pairs"].append(
                    {
                        "station_1_name": station_1["charging_station_name"],
                        "station_2_name": station_2["charging_station_name"],
                        "distance": d,
                    }
                )

        json_data["dist_to_goal"][station_1["charging_station_name"]] = int(
            distance(station_1["lat_long"], dest_lat_long).km
        )
        print(f"{i} / {len(json_data['charger_loc_data'])}")

    with open(SUPERCHARGER_LOC_DATA, "w") as file:
        json.dump(json_data, file, indent=4)


def convert_csv_to_json():
    query_positions()
    compute_distances()


def draw_graph(g: nx.Graph):
    edges = [(u, v) for (u, v, _) in g.edges(data=True)]

    # nodes
    pos = nx.get_node_attributes(g, "lat_long")
    nx.draw_networkx_nodes(g, pos, node_size=2)

    # edges
    nx.draw_networkx_edges(
        g, pos, edgelist=edges, width=0.05, alpha=0.5, edge_color="b", style="dashed"
    )

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.title("US Tesla Supercharger Network")
    plt.show()


def draw_shortest_path(g: nx.Graph, path_coords: np.ndarray, path_len: float, alg_name):
    # nodes
    pos = nx.get_node_attributes(g, "lat_long")
    nx.draw_networkx_nodes(g, pos, node_size=2)
    plt.plot(path_coords[:, 0], path_coords[:, 1], "g-")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.suptitle(f"Shortest Path from Palo Alto to Boston ({alg_name})")
    plt.title(f"Distance: {path_len} km", fontsize=10)
    plt.tight_layout()
    plt.show()


def calc_simple_shortest_path(
    g: nx.Graph, algorithm_name: str, loc_lookup: Dict[str, List[float]]
):
    shortest_path = nx.shortest_path(
        g,
        source=START_STATION,
        target=END_STATION,
        weight="weight",
        method=algorithm_name.lower(),
    )
    shortest_path_len = nx.shortest_path_length(
        g,
        source=START_STATION,
        target=END_STATION,
        weight="weight",
        method=algorithm_name.lower(),
    )
    path_locs = np.array([loc_lookup[station_name] for station_name in shortest_path])

    draw_shortest_path(g, path_locs, shortest_path_len, algorithm_name)


def calc_shortest_path_with_heuristic(
    g: nx.Graph,
    loc_lookup: Dict[str, List[float]],
    dist_lookup: Dict[str, int],
):
    def astar_heuristic(node_1: str, node_2: str):
        if node_2 != END_STATION:
            raise ValueError(f"Node 2 should always be {END_STATION}")
        return dist_lookup[node_1]

    shortest_path = nx.astar_path(
        g,
        source=START_STATION,
        target=END_STATION,
        heuristic=astar_heuristic,
        weight="weight",
    )
    shortest_path_len = nx.astar_path_length(
        g,
        source=START_STATION,
        target=END_STATION,
        heuristic=astar_heuristic,
        weight="weight",
    )

    path_locs = np.array([loc_lookup[station_name] for station_name in shortest_path])

    draw_shortest_path(g, path_locs, shortest_path_len, "A*")


def analyze_graph():
    g = nx.Graph()

    with open(SUPERCHARGER_LOC_DATA, "r") as file:
        data = json.load(file)
        charger_loc_data = data["charger_loc_data"]
        dist_data = data["dist_pairs"]
        dist_to_goal_lookup = data["dist_to_goal"]
    for i in range(len(charger_loc_data)):
        g.add_node(
            charger_loc_data[i]["charging_station_name"],
            lat_long=(
                charger_loc_data[i]["lat_long"][1],
                charger_loc_data[i]["lat_long"][0],
            ),
        )
    for pair in dist_data:
        g.add_edge(
            pair["station_1_name"], pair["station_2_name"], weight=pair["distance"]
        )

    draw_graph(g)

    # Create dictionary mapping from supercharger name to long-lat
    loc_lookup = {
        loc["charging_station_name"]: [loc["lat_long"][1], loc["lat_long"][0]]
        for loc in charger_loc_data
    }

    calc_simple_shortest_path(g, "Dijkstra", loc_lookup)
    calc_simple_shortest_path(g, "Bellman-Ford", loc_lookup)
    calc_shortest_path_with_heuristic(g, loc_lookup, dist_to_goal_lookup)


if __name__ == "__main__":
    # convert_csv_to_json()
    analyze_graph()
