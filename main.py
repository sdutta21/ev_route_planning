from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Tuple, Dict
import json

from geopy.geocoders import Nominatim
# https://geopy.readthedocs.io/en/stable/#module-geopy.distance
from geopy.distance import geodesic
from geopy.exc import GeocoderUnavailable

GEOLOCATOR = Nominatim(user_agent="supercharger_locator")


@dataclass
class SuperchargerData:
    station_name: str
    city: str
    state: str
    zip_code: int
    street_address: str

    @property
    def lat_long(self) -> Tuple[float, float]:
        # From https://geopy.readthedocs.io/en/stable/#module-geopy.geocoders
        location = GEOLOCATOR.geocode(self.street_address)
        return (location.latitude, location.longitude)

    @property
    def as_dict(self) -> Dict:
        dict = self.__dict__
        dict.update({"lat_long": self.lat_long})
        return dict


def convert_csv_to_json():
    json_data = {"data": []}
    data_headers = None
    with open(str(Path(__file__).parent / "data/tesla_superchargers.csv")) as file:
        csvreader = csv.reader(file, delimiter=",")
        for row in csvreader:
            # Get name of each column from CSV (first row)
            if not data_headers:
                data_headers = row
                continue

            charger_data = SuperchargerData(
                *{data_headers[i]: row[i] for i in range(len(row))}
            )

            # Catching 2 Different errors here:
            #   1. Attribute errors for when GeoPy fails to find a location for the inputted address
            #   2. GeocoderUnavailable for when GeoPy times out when searching for the inputted location
            try:
                json_data["data"].append(charger_data.as_dict)
            except (AttributeError, GeocoderUnavailable):
                continue

    with open(
        str(Path(__file__).parent / "data/tesla_superchargers.json"), "w"
    ) as file:
        json.dump(json_data, file, indent=4)


if __name__ == "__main__":
    convert_csv_to_json()
