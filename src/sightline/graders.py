"""
Interface for graders, which evaluate the output of models (against ground truth, if applicable).

JSON parsing is intentionally fuzzy to account for variations in output.
"""
import re
import json
from abc import ABC, abstractmethod


class Grader(ABC):
    @abstractmethod
    def grade(self, model_output_csv, ground_truth_csv = None) -> None:
        pass

    @staticmethod
    def response_to_JSON_object(response):
        """Extract JSON list as Python object from full response."""
        try:
            raw_response = str(response)
                
            potential_json = ""
            if ("{" in raw_response) and ("}" in raw_response):
                _, trim_beginning = raw_response.split("{", maxsplit=1)
                trim_end, _ = trim_beginning.rsplit("}", maxsplit=1)
                potential_json = trim_end.strip()
                potential_json = "{" + potential_json + "}"
                try:
                    json.loads(potential_json)
                except:
                    # when strings are not formatted with quotes
                    s = re.sub(r'([{,]\s*)([A-Za-z_]\w*)(\s*:)', r'\1"\2"\3', potential_json)  # Quote unquoted keys
                    s = s.replace("'", '"')  # Convert single quotes to double quotes
                    potential_json = s
            json_object = json.loads(potential_json)  # ensure valid JSON
            return json_object
        except Exception as e:
            print(e)
            return {}
    
    @staticmethod
    def response_to_JSON_list(response):  # future: fold into response_to_JSON_object
        """Extract JSON list as Python object from full response."""
        try:
            raw_response = str(response)
                
            potential_json = ""
            if ("[" in raw_response) and ("]" in raw_response):
                _, trim_beginning = raw_response.split("[", maxsplit=1)
                trim_end, _ = trim_beginning.rsplit("]", maxsplit=1)
                potential_json = trim_end.strip()
                potential_json = "[" + potential_json + "]"
                try:
                    json.loads(potential_json)
                except:
                    # when strings are not formatted with quotes
                    s = re.sub(r'([{,]\s*)([A-Za-z_]\w*)(\s*:)', r'\1"\2"\3', potential_json)  # Quote unquoted keys
                    s = s.replace("'", '"')  # Convert single quotes to double quotes
                    potential_json = s
            json_list = json.loads(potential_json)  # ensure valid JSON
            return json_list
        except Exception as e:
            print(e)
            return []