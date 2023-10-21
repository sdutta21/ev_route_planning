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


def convert_csv_to_json():
    json_data = {"charger_loc_data": [], "dist_pairs": []}
    data_headers = None
    counter = 0
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
            print(counter)
            counter += 1

    for i in range(len(json_data["charger_loc_data"])):
        for j in range(i + 1, len(json_data["charger_loc_data"])):
            station_1 = json_data["charger_loc_data"][i]
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

    with open(SUPERCHARGER_LOC_DATA, "w") as file:
        json.dump(json_data, file, indent=4)


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
    plt.show()


def draw_shortest_path(g: nx.Graph, path_coords: np.ndarray, path_len: float, alg_name):
    # nodes
    pos = nx.get_node_attributes(g, "lat_long")
    nx.draw_networkx_nodes(g, pos, node_size=2)
    plt.plot(path_coords[:, 0], path_coords[:, 1], "g-")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.title(f"{alg_name} Shortest Path (dist: {path_len} km)")
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


def calc_shortest_path_with_heuristic():
    pass


def analyze_graph():
    g = nx.Graph()

    with open(SUPERCHARGER_LOC_DATA, "r") as file:
        data = json.load(file)
        charger_loc_data = data["charger_loc_data"]
        dist_data = data["dist_pairs"]
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


if __name__ == "__main__":
    # convert_csv_to_json()
    analyze_graph()
