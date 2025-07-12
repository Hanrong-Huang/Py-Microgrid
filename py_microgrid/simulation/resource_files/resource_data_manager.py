"""
Resource data management utilities for py_microgrid.
Handles downloading and managing solar and wind resource data files.
"""

import os
from typing import Dict, Optional
import requests
import pandas as pd
import io

class ResourceDataManager:
    """Handles downloading and managing solar and wind resource data."""
    
    def __init__(self, api_key: str, email: str, 
                 solar_dir: Optional[str] = None,
                 wind_dir: Optional[str] = None):
        """
        Initialize ResourceDataManager.
        Uses existing HOPP resource directories without creating new ones.
        
        Args:
            api_key: API key for accessing NREL data
            email: User email for authentication
            solar_dir: Optional custom directory for solar data files
            wind_dir: Optional custom directory for wind data files
        """
        self.api_key = api_key
        self.email = email
        
        # Use existing HOPP resource directories
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.solar_dir = solar_dir or os.path.join(package_dir, 'simulation/resource_files/solar')
        self.wind_dir = wind_dir or os.path.join(package_dir, 'simulation/resource_files/wind')
        
        # Verify directories exist
        if not os.path.exists(self.solar_dir) or not os.path.exists(self.wind_dir):
            raise ValueError("Resource directories not found in HOPP package structure")

    def _get_existing_file(self, directory: str, exact_filename: str) -> Optional[str]:
        """
        Check for existing file with exact filename.
        
        Args:
            directory: Directory to search in
            exact_filename: Exact filename to match
            
        Returns:
            Optional[str]: Path to existing file if found, None otherwise
        """
        file_path = os.path.join(directory, exact_filename)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return file_path
        return None

    def _download_nasa_solar_data(self, latitude: float, longitude: float, year: str, file_path: str) -> str:
        """
        Downloads solar data from NASA Power API for a full year and formats it for HOPP compatibility. This is for solar data after 2020.
        
        Args:
            latitude: Site latitude
            longitude: Site longitude
            year: Year for solar data (full year)
            file_path: Path to save the file
        """
        nasa_api_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        params = {
            "start": f"{year}0101",
            "end": f"{year}1231",
            "latitude": latitude,
            "longitude": longitude,
            "community": "RE",
            "parameters": "ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,ALLSKY_SFC_SW_DWN,T2M,WS50M",
            "format": "CSV",
            "header": "true",
            "time-standard": "LST"
        }
        
        response = requests.get(nasa_api_url, params=params)
        
        if response.status_code == 200:
            # Find the start of the data section
            content = response.text
            data_start_index = content.find("YEAR,MO,DY,HR,")
            if data_start_index == -1:
                raise RuntimeError("Could not find the start of the data section in NASA Power response.")
            
            csv_data = content[data_start_index:]
            
            # Read the data into a pandas DataFrame
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Rename columns to match Himawari format
            df.rename(columns={
                'YEAR': 'Year',
                'MO': 'Month',
                'DY': 'Day',
                'HR': 'Hour',
                'ALLSKY_SFC_SW_DNI': 'DNI',
                'ALLSKY_SFC_SW_DIFF': 'DHI',
                'ALLSKY_SFC_SW_DWN': 'GHI',
                'T2M': 'Temperature',
                'WS50M': 'Wind Speed'
            }, inplace=True)
            
            # Add missing columns with default values
            df['Minute'] = 0
            df['Dew Point'] = 0
            df['Pressure'] = 0
            df['Wind Direction'] = 0
            df['Surface Albedo'] = 0
            
            # Reorder columns to match Himawari data format (no Time Zone in data section)
            himawari_data_columns = [
                'Year', 'Month', 'Day', 'Hour', 'Minute', 
                'DNI', 'DHI', 'GHI', 'Dew Point', 'Temperature', 
                'Pressure', 'Wind Direction', 'Wind Speed', 'Surface Albedo'
            ]
            df = df[himawari_data_columns]
            
            # Calculate metadata values
            time_zone = round(longitude / 15)
            elevation = 606 if -37 < latitude < -33 and 148 < longitude < 151 else 200  # Canberra area or default
            location_id = f"{abs(latitude):.0f}{abs(longitude):.0f}"
            
            # Create simplified header lines 
            header_line1 = "Source,Location ID,Latitude,Longitude,Time Zone,Elevation,DNI Units,DHI Units,GHI Units,Temperature Units"
            header_line2 = f"NASA Power,{location_id},{latitude},{longitude},{time_zone},{elevation},w/m2,w/m2,w/m2,c"
            
            # Write file in Himawari format that HOPP expects
            with open(file_path, 'w') as f:
                # Write metadata headers
                f.write(header_line1 + '\n')
                f.write(header_line2 + '\n')
                
                # Write data with column headers
                df.to_csv(f, index=False)
            
            print(f"Solar data downloaded from NASA Power for year {year} and saved to {file_path}.")
            print(f"Metadata: Time Zone {time_zone}, Elevation {elevation}m, Location ID {location_id}")
            return file_path
        else:
            raise RuntimeError(f"Failed to download solar data from NASA Power: {response.status_code}\n{response.text}")

    def download_solar_data(self, latitude: float, longitude: float, year: str) -> str:
        """
        Get solar resource data from Himawari for a full year, first trying existing file then downloading if needed. This is for solar data before 2020.
        
        Note: Both Himawari (≤2020) and NASA Power (>2020) provide full-year datasets only.
        
        Args:
            latitude: Site latitude
            longitude: Site longitude
            year: Year for solar data (full year only)
            
        Returns:
            str: Path to solar data file
            
        Raises:
            RuntimeError: If can't get data and no existing file found
        """
        # Generate exact filename
        filename = f"{latitude}_{longitude}_psmv3_60_{year}.csv"
        file_path = os.path.join(self.solar_dir, filename)
        
        # Check for existing file with exact coordinates
        existing_file = self._get_existing_file(self.solar_dir, filename)
        if existing_file:
            print(f"Using existing solar data file: {existing_file}")
            return existing_file
        
        if int(year) <= 2020:
            # If no existing file, try to download from Himawari (full year only)
            try:
                solar_base_url = "https://developer.nrel.gov/api/nsrdb/v2/solar/himawari-download.csv"
                solar_params = {
                    "wkt": f"POINT({longitude} {latitude})",
                    "names": year,
                    "leap_day": "false",
                    "interval": "60",
                    "utc": "false",
                    "full_name": "Hanrong Huang",
                    "email": self.email,
                    "affiliation": "UNSW",
                    "mailing_list": "true",
                    "reason": "research",
                    "api_key": self.api_key,
                    "attributes": "dni,dhi,ghi,dew_point,air_temperature,surface_pressure,wind_direction,wind_speed,surface_albedo"
                }
                
                response = requests.get(solar_base_url, params=solar_params)
                
                if response.status_code == 200:
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                    print(f"Solar data downloaded from Himawari for year {year} and saved to {file_path}.")
                    return file_path
                else:
                    # If download failed, check one more time for exact coordinate file
                    existing_file = self._get_existing_file(self.solar_dir, filename)
                    if existing_file:
                        print(f"Download failed, using existing file: {existing_file}")
                        return existing_file
                    raise RuntimeError(f"Failed to download solar data: {response.status_code}\n{response.text}")
            except Exception as e:
                # Final check for existing file before giving up
                existing_file = self._get_existing_file(self.solar_dir, filename)
                if existing_file:
                    print(f"Download failed, using existing file: {existing_file}")
                    return existing_file
                raise RuntimeError(f"Failed to get solar data: {str(e)}")
        else:
            # Year is after 2020, download from NASA Power (full year)
            return self._download_nasa_solar_data(latitude, longitude, year, file_path)

    def download_wind_data(self, latitude: float, longitude: float, 
                          start_date: str, end_date: str) -> str:
        """
        Get wind resource data, first trying existing file then downloading if needed.
        
        Args:
            latitude: Site latitude
            longitude: Site longitude
            start_date: Start date for wind data (format: YYYYMMDD)
            end_date: End date for wind data (format: YYYYMMDD)
            
        Returns:
            str: Path to wind data file
            
        Raises:
            RuntimeError: If can't get data and no existing file found
        """
        # Generate exact filename
        filename = f"{latitude}_{longitude}_NASA_{start_date[:4]}_60min_50m.srw"
        file_path = os.path.join(self.wind_dir, filename)
        
        # Check for existing file with exact coordinates
        existing_file = self._get_existing_file(self.wind_dir, filename)
        if existing_file:
            print(f"Using existing wind data file: {existing_file}")
            return existing_file
        
        # If no existing file, try to download
        try:
            wind_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
            wind_params = {
                "start": start_date,
                "end": end_date,
                "latitude": latitude,
                "longitude": longitude,
                "community": "ag",
                "parameters": "WS50M,WD50M",
                "format": "srw",
                "user": "Hanrong",
                "header": "true",
                "time-standard": "lst"
            }
            
            response = requests.get(wind_url, params=wind_params)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"Wind data downloaded successfully and saved to {file_path}.")
                return file_path
            else:
                # If download failed, check one more time for exact coordinate file
                existing_file = self._get_existing_file(self.wind_dir, filename)
                if existing_file:
                    print(f"Download failed, using existing file: {existing_file}")
                    return existing_file
                raise RuntimeError(f"Failed to download wind data: {response.status_code}\n{response.text}")
        except Exception as e:
            # Final check for existing file before giving up
            existing_file = self._get_existing_file(self.wind_dir, filename)
            if existing_file:
                print(f"Download failed, using existing file: {existing_file}")
                return existing_file
            raise RuntimeError(f"Failed to get wind data: {str(e)}")
