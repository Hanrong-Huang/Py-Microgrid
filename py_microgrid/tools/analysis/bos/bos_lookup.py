from scipy.interpolate import LinearNDInterpolator as interp
from pathlib import Path
import pandas as pd
import numpy as np

from .bos_model import BOSCalculator
from py_microgrid.utilities.log import bos_logger as logger

file_path = Path(__file__).parent


class BOSLookup(BOSCalculator):
    def __init__(self):
        super().__init__()
        self.name = "BOSLookup"

        self.input_parameters = ["Interconnection Capacity",
                                 "Wind Installed Capacity",
                                 "Solar Installed Capacity"]

        # List of desired output parameters from the JSON lookup
        self.desired_output_parameters = ["Wind BOS Cost",
                                          "Solar BOS Cost"]

        # Loads the json data containing all the BOS cost information from the excel model
        self.data, self.contents = self._load_lookup()
        self.interpolating_fxns = self._load_interp()

        for p in self.desired_output_parameters:
            if p not in self.data.columns:
                raise KeyError(p + " column missing")

    def _load_lookup(self):
        file = file_path / "BOSLookup.csv"
        with open(file, "r") as f:
            data = pd.read_csv(f)
        contents = data[self.input_parameters].values
        return data, contents

    def _load_interp(self):
        fxns = []
        for p in self.desired_output_parameters:
            f = interp(self.contents, self.data[p].values)
            fxns.append(f)
        return fxns

    def _lookup_costs(self, wind_mw, solar_mw, interconnection_mw):
        if wind_mw + solar_mw == 0:
            return 0, 0, 0

        search_inputs = np.array([interconnection_mw, wind_mw, solar_mw])
        distance_norm = np.linalg.norm(self.contents - search_inputs, axis=1)
        min_index = np.argmin(distance_norm)
        min_distance = distance_norm[min_index]

        vals = []
        for i in range(len(self.desired_output_parameters)):
            vals.append(self.interpolating_fxns[i](search_inputs)[0])

        if np.isnan(vals).any():
            if min_distance / np.linalg.norm(search_inputs) < .05:
                # Close match found - use nearest neighbor
                wind_bos_cost = self.data.iloc[min_index:min_index+1]["Wind BOS Cost"].values[0]
                solar_bos_cost = self.data.iloc[min_index:min_index+1]["Solar BOS Cost"].values[0]
            else:
                # Interpolation failed - use intelligent extrapolation
                wind_bos_cost, solar_bos_cost = self._extrapolate_costs(wind_mw, solar_mw, interconnection_mw)
        else:
            wind_bos_cost = vals[self.desired_output_parameters.index("Wind BOS Cost")]
            solar_bos_cost = vals[self.desired_output_parameters.index("Solar BOS Cost")]

        total_bos_cost = wind_bos_cost + solar_bos_cost
        logger.info("Total BOS Cost: {} Wind BOS Cost: {} Solar BOS Cost {}".
                    format(total_bos_cost, wind_bos_cost, solar_bos_cost))

        return wind_bos_cost, solar_bos_cost, total_bos_cost, min_distance

    def _extrapolate_costs(self, wind_mw, solar_mw, interconnection_mw):
        """
        Intelligently extrapolate costs when interpolation fails.
        Uses linear extrapolation based on nearby points in the lookup table.
        """
        logger.info(f"Extrapolating BOS costs for Wind={wind_mw}MW, Solar={solar_mw}MW, Interconnection={interconnection_mw}MW")
        
        # Find points with similar interconnection capacity
        interconnection_tolerance = 50  # MW
        similar_interconnection = self.data[
            abs(self.data['Interconnection Capacity'] - interconnection_mw) <= interconnection_tolerance
        ]
        
        if len(similar_interconnection) == 0:
            # No similar interconnection - use global scaling
            return self._global_scaling_extrapolation(wind_mw, solar_mw, interconnection_mw)
        
        # Separate wind and solar cost extrapolation
        wind_bos_cost = self._extrapolate_wind_cost(wind_mw, similar_interconnection)
        solar_bos_cost = self._extrapolate_solar_cost(solar_mw, similar_interconnection)
        
        logger.info(f"Extrapolated costs: Wind BOS=${wind_bos_cost:.2f}, Solar BOS=${solar_bos_cost:.2f}")
        return wind_bos_cost, solar_bos_cost

    def _extrapolate_wind_cost(self, wind_mw, data_subset):
        """Extrapolate wind BOS cost based on similar interconnection capacity data."""
        if wind_mw == 0:
            return 0
        
        # Find data points with wind capacity
        wind_data = data_subset[data_subset['Wind Installed Capacity'] > 0]
        if len(wind_data) < 2:
            # Not enough data for extrapolation - use simple scaling
            avg_cost_per_mw = wind_data['Wind BOS Cost'].mean() / wind_data['Wind Installed Capacity'].mean()
            return wind_mw * avg_cost_per_mw
        
        # Linear extrapolation based on wind capacity
        wind_capacities = wind_data['Wind Installed Capacity'].values
        wind_costs = wind_data['Wind BOS Cost'].values
        
        # Use linear fit to extrapolate
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(wind_capacities, wind_costs)
        extrapolated_cost = slope * wind_mw + intercept
        
        # Ensure positive result
        return max(0, extrapolated_cost)

    def _extrapolate_solar_cost(self, solar_mw, data_subset):
        """Extrapolate solar BOS cost based on similar interconnection capacity data."""
        if solar_mw == 0:
            return 0
        
        # Find data points with solar capacity
        solar_data = data_subset[data_subset['Solar Installed Capacity'] > 0]
        if len(solar_data) < 2:
            # Not enough data for extrapolation - use simple scaling
            avg_cost_per_mw = solar_data['Solar BOS Cost'].mean() / solar_data['Solar Installed Capacity'].mean()
            return solar_mw * avg_cost_per_mw
        
        # Linear extrapolation based on solar capacity
        solar_capacities = solar_data['Solar Installed Capacity'].values
        solar_costs = solar_data['Solar BOS Cost'].values
        
        # Use linear fit to extrapolate
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(solar_capacities, solar_costs)
        extrapolated_cost = slope * solar_mw + intercept
        
        # Ensure positive result
        return max(0, extrapolated_cost)

    def _global_scaling_extrapolation(self, wind_mw, solar_mw, interconnection_mw):
        """
        Fallback extrapolation using global scaling factors from the entire dataset.
        """
        logger.info("Using global scaling extrapolation as fallback")
        
        # Calculate average cost per MW for each technology
        wind_data = self.data[self.data['Wind Installed Capacity'] > 0]
        solar_data = self.data[self.data['Solar Installed Capacity'] > 0]
        
        if len(wind_data) > 0:
            avg_wind_cost_per_mw = wind_data['Wind BOS Cost'].mean() / wind_data['Wind Installed Capacity'].mean()
        else:
            avg_wind_cost_per_mw = 1000000  # Default fallback
        
        if len(solar_data) > 0:
            avg_solar_cost_per_mw = solar_data['Solar BOS Cost'].mean() / solar_data['Solar Installed Capacity'].mean()
        else:
            avg_solar_cost_per_mw = 1000000  # Default fallback
        
        wind_bos_cost = wind_mw * avg_wind_cost_per_mw
        solar_bos_cost = solar_mw * avg_solar_cost_per_mw
        
        return wind_bos_cost, solar_bos_cost

    def calculate_bos_costs(self, wind_mw, solar_mw, interconnection_mw, scenario='greenfield'):
        """
        Calls the appropriate calculate_bos_costs_x method for the Cost Source data specified

        :param wind_mw: Installed Capacity (MW) of wind component
        :param solar_mw: Installed Capacity (MW) of solar component
        :param interconnection_mw:
        :param scenario: 'greenfield' or 'solar addition'
        :return: wind, solar and total bos cost
        """
        scenario = scenario.lower()
        if scenario == 'greenfield':
            return self._lookup_costs(wind_mw, solar_mw, interconnection_mw)
        elif scenario == 'solar addition':
            raise NotImplementedError
        else:
            raise ValueError("scenario type {} not recognized".format(scenario))
