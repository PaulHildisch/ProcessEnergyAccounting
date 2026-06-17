import requests
from requests.auth import HTTPBasicAuth
from typing import Optional, List, Dict, Any

# Map sensor type codes to human-readable names
SENSOR_TYPE_MAP = {
    1: "Line power meter",
    9: "Line power meter with residual current",
    8: "Outlet power meter",
    7: "Digital Inputs",
    12: "Bender RCMB Module",
    20: "System Data (sensor group)",
    51: "Temperature Sensor",
    52: "Temperature/Humidity Sensor",
    53: "Temperature/Humidity/AirPressure Sensor",
    101: "Bank (eFuses Port-groups) Sensor",
    102: "DC Power Sources",
}

class SmartMeterAPIClient:
    """
    Client for fetching and querying smart meter data via HTTP JSON.

    Attributes:
        base_url (str): The base URL for the API endpoint.
        timeout (int): Timeout for HTTP requests in seconds.
        auth (Optional[HTTPBasicAuth]): HTTP Basic Authentication object.
    """
    # Component flags
    DESCR = 0x10000  # sensor_descr
    VALUES = 0x4000  # sensor_values
    EXTENDED = 0x800000  # enables complex groups

    def __init__(
        self,
        host: str,
        ssl: bool = True,
        timeout: int = 10,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
                Initialize the SmartMeterAPIClient.

                Args:
                    host (str): The hostname or IP address of the smart meter.
                    ssl (bool): Whether to use HTTPS (default: True).
                    timeout (int): Timeout for HTTP requests in seconds (default: 10).
                    username (Optional[str]): Username for HTTP Basic Authentication (default: None).
                    password (Optional[str]): Password for HTTP Basic Authentication (default: None).
                """
        scheme = 'https' if ssl else 'http'
        self.base_url = f"{scheme}://{host}/status.json"
        self.timeout = timeout
        self.auth = HTTPBasicAuth(username, password) if username else None

    def _fetch(self, skip_complex: bool = False, skip_simple: bool = False) -> Dict[str, Any]:
        """
        Internal method to fetch raw JSON data from the device.

        Args:
            skip_complex (bool): If True, skip fetching complex sensor groups (default: False).
            skip_simple (bool): If True, skip fetching simple sensors (default: False).

        Returns:
            Dict[str, Any]: The JSON response from the device.
        """
        components = self.DESCR + self.VALUES
        if skip_complex:
            params = {'components': components}
        elif skip_simple:
            params = {'components': components + self.EXTENDED, 'types': 'C'}
        else:
            params = {'components': components + self.EXTENDED}

        resp = requests.get(
            self.base_url,
            params=params,
            verify=False,
            auth=self.auth,
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def list_sensors(self) -> List[Dict[str, Any]]:
        """
        List all sensors with their type, id, and name.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing sensor details.
        """
        data = self._fetch()
        sensors = []
        for d in data.get('sensor_descr', []):
            t = d.get('type')
            for prop in d.get('properties', []):
                sensors.append({
                    'type_code': t,
                    'type_name': SENSOR_TYPE_MAP.get(t, 'Unknown'),
                    'id': prop.get('id'),
                    'name': prop.get('name')
                })
        return sensors

    def get_sensor_data(self) -> List[Dict[str, Any]]:
        """
        Retrieve sensor readings, including both flat and grouped sensors.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing sensor readings.
        """
        data = self._fetch()
        descr_map = {d['type']: d for d in data.get('sensor_descr', [])}
        readings: List[Dict[str, Any]] = []

        for entry in data.get('sensor_values', []):
            t = entry.get('type')
            descr = descr_map.get(t, {})
            props = descr.get('properties', [])

            if 'fields' in descr:
                fields = descr['fields']
                for idx, prop in enumerate(props):
                    raw_vals = entry['values'][idx]
                    readings.append({
                        'id': prop['id'],
                        'name': prop.get('name'),
                        'type_code': t,
                        'type_name': SENSOR_TYPE_MAP.get(t, 'Unknown'),
                        'data': {f['name']: raw_vals[i].get('v') for i, f in enumerate(fields)}
                    })

            elif 'groups' in descr:
                groups = descr['groups']
                for prop_idx, prop in enumerate(props):
                    flat: Dict[str, Any] = {}
                    prop_values = entry['values'][prop_idx]
                    for g_idx, group in enumerate(groups):
                        group_values = prop_values[g_idx]
                        if not group_values:
                            continue
                        instance_vals = group_values[0]
                        for f_idx, field in enumerate(group['fields']):
                            if f_idx < len(instance_vals):
                                flat[field['name']] = instance_vals[f_idx].get('v')
                            else:
                                flat[field['name']] = None
                    readings.append({
                        'id': prop['id'],
                        'name': prop.get('name'),
                        'type_code': t,
                        'type_name': SENSOR_TYPE_MAP.get(t, 'Unknown'),
                        'data': flat
                    })

        return readings

    def get_field(self, node: str, field: str) -> Any:
        """
        Fetch a specific field value for the given node id or name.

        Args:
            node (str): The sensor node id or name.
            field (str): The field name to retrieve.

        Returns:
            Any: The value of the specified field.

        Raises:
            KeyError: If the node or field is not found.
        """
        for sensor in self.get_sensor_data():
            if sensor['id'] == node or sensor['name'] == node:
                if field in sensor['data']:
                    return sensor['data'][field]
                raise KeyError(f"Field '{field}' not found for node '{node}'")
        raise KeyError(f"Sensor node '{node}' not found")

    def get_power_usage(self, node: str, power_type: str = 'ActivePower') -> float:
        """
        Convenience method to get the power usage of a node.

        Args:
            node (str): The sensor node id or name.
            power_type (str): The type of power to retrieve (default: 'ActivePower').

        Returns:
            float: The power usage value.
        """
        return self.get_field(node, power_type)