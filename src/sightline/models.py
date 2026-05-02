"""
Interfaces to language models. 

Install optional dependencies for specific models using `pip install sightline[ollama]` or `pip install "your-package[ollama,openai]"`.
(Note that currently, additional code for ollama and openai are available.)
"""
import os
import csv
import base64
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    @abstractmethod
    def __init__(self, model_type: str, timeout_sec=60, pool_size=1):
        self._model_type: str = model_type  # future: make actual model instead of name
        self._timeout_sec = timeout_sec
        self._pool_size: int = pool_size
        # future: move the rest of the multiton/client/pool logic to Model class

    @property
    def model_type(self) -> str:
        return self._model_type
    
    @property
    def timeout_sec(self):
        return self._timeout_sec
    
    @property
    def pool_size(self) -> int:
        return self._pool_size
    
    def run_test(self, input_csv_path: str, id_column: str, user_prompt: str, output_csv_path: str, \
                    system_prompt: str = None, temperature: float = None, start_at_index: int = 0, end_before_index: int = None, \
                    image_column: str = None, image_delimiter: str = ",", repeated_text_only: bool = False) -> None:
        """Given user_prompt / system_prompt and paths to per-item image input, send test question(s) to the model, 
        and write results to output_csv_path.

        """
        rows = []
        raw_table = pd.read_csv(input_csv_path, encoding='utf-8')
        if (id_column not in raw_table.columns):  # future: auto-numbering
            raise Exception("id_column not found in fieldnames")
        else:
            if repeated_text_only:  # an edge case
                no_dups_table = raw_table[[id_column]]
                no_dups_table = no_dups_table.drop_duplicates()
                rows = no_dups_table.to_dict("records")
            else:
                if (image_column not in raw_table.columns):
                    raise Exception("image_column not found in fieldnames")
                else:  # average case
                    no_dups_table = raw_table[[id_column, image_column]]
                    no_dups_table = no_dups_table.drop_duplicates()
                    rows = no_dups_table.to_dict("records")
            
            # Save data to CSV while running (no writer conflicts when sequential)
            file_is_new = not os.path.exists(output_csv_path)
            CSV_HEADERS = ['id', 'raw_response', 'request_start', 'request_end']
            with open(output_csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if file_is_new:
                    writer.writerow(CSV_HEADERS)
                    
                i = 0
                while i < len(rows):
                    if i < start_at_index:
                        i += 1
                        continue
                    if end_before_index is not None and i >= end_before_index:
                        break

                    if repeated_text_only:
                        image_list = []
                    else:
                        paths = str(rows[i][image_column]).split(sep=image_delimiter)
                        image_list: list[str] = self._open_images(paths)
                    
                    # send request and receive raw response (will clean response at Grader)
                    raw_response, start_datetime, end_datetime = self.basic_single_request( \
                        system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature, \
                        image_list=image_list)
                    # print(f"{i} is complete")
                    
                    # Write the list as a new row to the CSV file  ## future: unpack the given metadata
                    current_row = [rows[i][id_column], raw_response, start_datetime, end_datetime]
                    writer.writerow(current_row)

                    i += 1
                
    @abstractmethod
    def basic_single_request(self, user_prompt, system_prompt=None, temperature=None, image_list=[], complete_res_to_stdout: bool = False):
        pass
    
    @staticmethod
    def _open_images(paths):
        return [Model._encode_image(path) for path in paths]

    @staticmethod
    def _encode_image(image_path):
        """Encode image in Base 64 format."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


class OllamaModel(Model):
    from ollama import Client
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool: list = []

        # initialize one client
        ollama_client = self.Client(host='http://localhost:11434', timeout=super().timeout_sec)
        self._pool.append({'client': ollama_client, 'busy': 0})
    
    def basic_single_request(self, user_prompt: str, system_prompt: str = None, temperature: float = None, image_list=[], complete_res_to_stdout: bool = False):
        # build request body (messages)
        messages = []
        user_message = {  # should I do message validation?
            'role': 'user',
            'content': user_prompt
        }
        if image_list is not None and len(image_list) > 0:
            user_message.update({'images': image_list})
        messages.append(user_message)
        if system_prompt is not None:
            system_message = {
                'role': 'system',
                'content': system_prompt
            }
            messages.append(system_message)
        
        # set options
        options={}
        if temperature is not None:
            options.update({'temperature': float(temperature)})

        # send basic request and receive response
        start_request = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            res = self._get_client().chat(model=super().model_type, messages=messages, options=options)
            end_request = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            end_request = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(e)
            return ("", start_request, end_request)

        # return response and approximate start/completion times (note that this is generally available from complete output)
        raw_response_text = str(res['message']['content'])
        if complete_res_to_stdout:
            print(res)
        
        return (raw_response_text, start_request, end_request)

    def _get_client(self) -> Client:  # should move this logic to single_request
        """Return a free client (TO BE IMPLEMENTED)."""
        return self._pool[0]['client']


class OpenAIModel(Model):
    from openai import OpenAI
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool = []

        # initialize one client
        openai_client = self.OpenAI()
        self._pool.append({'client': openai_client, 'busy': 0})

    def basic_single_request(self, user_prompt: str, system_prompt: str = None, temperature: float = None, image_list=[], complete_res_to_stdout: bool = False):
        # build request body
        messages = []
        user_messages = {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_prompt
                }
            ],
        }
        if image_list is not None and len(image_list) > 0:
            images = [{
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image}",
                        # "image_url": f"data:image/jpeg;base64,{image}",
                    } for image in image_list]
            user_messages.update({'content': user_messages['content'] + images})
        
        messages.append(user_messages)
        if system_prompt is not None:
            system_messages = {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt
                    },
                ],
            }
            messages.append(system_messages)

        # send request body and receive response (what about tools?)
        start_request_system = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res = self._get_client().responses.create(
                    model=super().model_type,
                    input=messages,
                    temperature=temperature
                )
        end_request_system = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # return response and approximate start/completion times (could also obtain from logs; see res)
        raw_response_text = str(res.output_text)
        if complete_res_to_stdout:
            print(res)
        
        return (raw_response_text, start_request_system, end_request_system)

    def _get_client(self) -> OpenAI:  # should move this logic to single_request
        """Return a free client (queues/flags to be implemented)."""
        return self._pool[0]['client']