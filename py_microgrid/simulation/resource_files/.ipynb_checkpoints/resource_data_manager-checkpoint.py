"""
Resource data management utilities for HOPP.
Handles downloading and managing solar and wind resource data files.
"""

import os
from typing import Dict, Optional
import requests

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
    
    def download_solar_data(self, latitude: float, longitude: float, year: str) -> str:
        """
        Get solar resource data, first trying existing file then downloading if needed.
        
        Args:
            latitude: Site latitude
            longitude: Site longitude
            year: Year for solar data
            
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
        
        # If no existing file, try to download
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
                print(f"Solar data downloaded and saved to {file_path}.")
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