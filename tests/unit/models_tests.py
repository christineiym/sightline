"""
Unit tests for sightline/models.py

Both ollama and openai packages are assumed to be installed.
Network calls are mocked throughout — no real requests are made.

run_test coverage:
  - id_column validation
  - repeated_text_only=True (text-only path, no image_column required)
  - repeated_text_only=False + image_column missing (raises)
  - repeated_text_only=False + image_column present (average case)
  - start_at_index / end_before_index (exclusive upper bound)
  - deduplication before the loop
  - CSV header written only for new files, not when appending
  - output row content and ordering

Run with: python -m pytest ./tests/unit/models_tests.py
(This allows pytest to pick up the optional dependencies by ensuring the current directory is in sys.path.)

AI use disclosure: This was written with assistance from Claude Sonnet 4.6.
"""
import base64
import csv
import os
import pytest
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

from sightline.models import Model, OllamaModel, OpenAIModel


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================

FAKE_RESPONSE_TEXT = "test response"
FAKE_START = "2024-01-01 00:00:00"
FAKE_END = "2024-01-01 00:00:01"


def make_csv(rows: list[dict], id_col="id", image_col=None) -> str:
    """Return a CSV string with the given rows."""
    cols = list(rows[0].keys()) if rows else [id_col]
    out = StringIO()
    writer = csv.DictWriter(out, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


@pytest.fixture
def ollama_model():
    """OllamaModel with a mocked Ollama client."""
    with patch("sightline.models.OllamaModel.Client") as MockClient:
        mock_client_instance = MagicMock()
        MockClient.return_value = mock_client_instance
        model = OllamaModel(model_type="llava", timeout_sec=30, pool_size=1)
        model._pool[0]["client"] = mock_client_instance
        yield model, mock_client_instance


@pytest.fixture
def openai_model():
    """OpenAIModel with a mocked OpenAI client."""
    with patch("sightline.models.OpenAIModel.OpenAI") as MockOpenAI:
        mock_client_instance = MagicMock()
        MockOpenAI.return_value = mock_client_instance
        model = OpenAIModel(model_type="gpt-4.1", timeout_sec=30, pool_size=1)
        model._pool[0]["client"] = mock_client_instance
        yield model, mock_client_instance


# ===========================================================================
# Model (abstract base)
# ===========================================================================

class TestModelAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Model()

    def test_subclass_must_implement_basic_single_request(self):
        class Incomplete(Model):
            def __init__(self):
                super().__init__("test-model")
        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_omitting_init_raises(self):
        """__init__ is abstract; subclass must call super().__init__."""
        class NoInit(Model):
            def basic_single_request(self, *a, **kw): ...
        with pytest.raises(TypeError):
            NoInit()


class TestModelEncodeImage:
    def test_encode_image_returns_base64_string(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n")
        result = Model._encode_image(str(img))
        assert result == base64.b64encode(b"\x89PNG\r\n").decode("utf-8")

    def test_encode_image_is_string(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"data")
        assert isinstance(Model._encode_image(str(img)), str)

    def test_open_images_encodes_each_path(self, tmp_path):
        paths = []
        for i, content in enumerate([b"a", b"b", b"c"]):
            p = tmp_path / f"img{i}.png"
            p.write_bytes(content)
            paths.append(str(p))
        result = Model._open_images(paths)
        expected = [base64.b64encode(c).decode("utf-8") for c in [b"a", b"b", b"c"]]
        assert result == expected

    def test_open_images_empty_list(self):
        assert Model._open_images([]) == []


# ===========================================================================
# Model.run_test
# Tested via OllamaModel to avoid duplicating CSV/IO logic for every subclass.
# ===========================================================================

class TestRunTest:
    """
    Tests for Model.run_test covering all branches:
      - id_column missing
      - repeated_text_only=True (text-only edge case)
      - repeated_text_only=False, image_column missing
      - repeated_text_only=False, image_column present (average case)
      - start_at_index skipping
      - end_before_index early exit (exclusive upper bound)
      - header written only for new output files
      - header not written when appending to existing output file
      - duplicate rows deduplicated before the loop
    """

    def _make_model(self):
        """Return an OllamaModel with a mocked client and preset chat response."""
        with patch("sightline.models.OllamaModel.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            model = OllamaModel(model_type="llava")
            model._pool[0]["client"] = mock_client
            mock_client.chat.return_value = {
                "message": {"content": FAKE_RESPONSE_TEXT}
            }
            return model, mock_client

    def _run(self, model, csv_in, csv_out, **kwargs):
        """Convenience wrapper with sensible defaults."""
        defaults = dict(
            input_csv_path=str(csv_in),
            id_column="id",
            user_prompt="describe this",
            output_csv_path=str(csv_out),
        )
        defaults.update(kwargs)
        model.run_test(**defaults)

    # --- id_column validation ---

    def test_raises_when_id_column_missing(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("other_col,value\n1,hello\n")
        with pytest.raises(Exception, match="id_column not found"):
            self._run(model, csv_in, tmp_path / "out.csv")

    # --- repeated_text_only=True branch ---

    def test_repeated_text_only_sends_one_request_per_unique_id(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        # Three rows but only two unique ids
        csv_in.write_text("id,text\n1,hello\n1,hello\n2,world\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "basic_single_request",
                          return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)) as mock_req:
            self._run(model, csv_in, csv_out, repeated_text_only=True)
        assert mock_req.call_count == 2

    def test_repeated_text_only_passes_empty_image_list(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,text\n1,hello\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "basic_single_request",
                          return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)) as mock_req:
            self._run(model, csv_in, csv_out, repeated_text_only=True)
        _, kwargs = mock_req.call_args
        assert kwargs.get("image_list") == [] or mock_req.call_args[0][3] == []

    def test_repeated_text_only_does_not_require_image_column(self, tmp_path):
        """image_column should be ignored when repeated_text_only=True."""
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,text\n1,hello\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "basic_single_request",
                          return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)):
            # Should not raise even though image_column is not in the CSV
            self._run(model, csv_in, csv_out, repeated_text_only=True)

    # --- repeated_text_only=False, image_column missing ---

    def test_raises_when_image_column_missing(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,text\n1,hello\n")
        with pytest.raises(Exception, match="image_column not found"):
            self._run(model, csv_in, tmp_path / "out.csv",
                      image_column="image")

    # --- repeated_text_only=False, image_column present (average case) ---

    def test_average_case_sends_one_request_per_unique_row(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        # Two unique (id, image) pairs plus one duplicate
        csv_in.write_text("id,image\n1,a.png\n2,b.png\n1,a.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=["fakebase64"]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)) as mock_req:
                self._run(model, csv_in, csv_out, image_column="image")
        assert mock_req.call_count == 2

    def test_average_case_passes_encoded_images(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=["encoded"]) as mock_open_imgs:
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)) as mock_req:
                self._run(model, csv_in, csv_out, image_column="image")
        called_image_list = mock_req.call_args[1].get("image_list") or mock_req.call_args[0][3]
        assert called_image_list == ["encoded"]

    def test_average_case_splits_image_paths_by_delimiter(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png|b.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]) as mock_open_imgs:
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)):
                self._run(model, csv_in, csv_out, image_column="image", image_delimiter="|")
        mock_open_imgs.assert_called_once_with(["a.png", "b.png"])

    # --- start_at_index ---

    def test_start_at_index_skips_early_rows(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n2,b.png\n3,c.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)) as mock_req:
                self._run(model, csv_in, csv_out, image_column="image", start_at_index=2)
        # Only row at index 2 (id=3) processed
        assert mock_req.call_count == 1

    def test_start_at_index_zero_processes_all(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n2,b.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)) as mock_req:
                self._run(model, csv_in, csv_out, image_column="image", start_at_index=0)
        assert mock_req.call_count == 2

    # --- end_before_index ---

    def test_end_before_index_is_exclusive(self, tmp_path):
        """end_before_index=1 should process only row 0, not row 1."""
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n2,b.png\n3,c.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)) as mock_req:
                self._run(model, csv_in, csv_out, image_column="image", end_before_index=1)
        assert mock_req.call_count == 1

    def test_end_before_index_none_processes_all(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n2,b.png\n3,c.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)) as mock_req:
                self._run(model, csv_in, csv_out, image_column="image", end_before_index=None)
        assert mock_req.call_count == 3

    def test_start_and_end_combined(self, tmp_path):
        """start_at_index=1, end_before_index=3 should process rows 1 and 2 only."""
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n2,b.png\n3,c.png\n4,d.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)) as mock_req:
                self._run(model, csv_in, csv_out, image_column="image",
                          start_at_index=1, end_before_index=3)
        assert mock_req.call_count == 2

    # --- CSV output ---

    def test_header_written_for_new_output_file(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n")
        csv_out = tmp_path / "out.csv"  # does not exist yet
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)):
                self._run(model, csv_in, csv_out, image_column="image")
        with open(csv_out) as f:
            reader = csv.reader(f)
            first_row = next(reader)
        assert first_row == ["id", "raw_response", "request_start", "request_end"]

    def test_header_not_written_when_appending(self, tmp_path):
        """Calling run_test on an existing output file should not add a second header."""
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n2,b.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)):
                # First call creates the file and writes the header
                self._run(model, csv_in, csv_out, image_column="image",
                          end_before_index=1)
                # Second call appends — should not write another header
                self._run(model, csv_in, csv_out, image_column="image",
                          start_at_index=1)
        with open(csv_out) as f:
            rows = list(csv.reader(f))
        header_rows = [r for r in rows if r == ["id", "raw_response", "request_start", "request_end"]]
        assert len(header_rows) == 1

    def test_output_row_contains_id_and_response(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n42,a.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=("my answer", FAKE_START, FAKE_END)):
                self._run(model, csv_in, csv_out, image_column="image")
        with open(csv_out) as f:
            rows = list(csv.reader(f))
        data_row = rows[1]  # rows[0] is header
        assert data_row[0] == "42"
        assert data_row[1] == "my answer"

    def test_output_row_contains_timestamps(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)):
                self._run(model, csv_in, csv_out, image_column="image")
        with open(csv_out) as f:
            rows = list(csv.reader(f))
        data_row = rows[1]
        assert data_row[2] == FAKE_START
        assert data_row[3] == FAKE_END

    def test_multiple_rows_written_in_order(self, tmp_path):
        model, _ = self._make_model()
        csv_in = tmp_path / "input.csv"
        csv_in.write_text("id,image\n1,a.png\n2,b.png\n3,c.png\n")
        csv_out = tmp_path / "out.csv"
        with patch.object(model, "_open_images", return_value=[]):
            with patch.object(model, "basic_single_request",
                              return_value=(FAKE_RESPONSE_TEXT, FAKE_START, FAKE_END)):
                self._run(model, csv_in, csv_out, image_column="image")
        with open(csv_out) as f:
            rows = list(csv.reader(f))
        assert [r[0] for r in rows[1:]] == ["1", "2", "3"]


# ===========================================================================
# OllamaModel
# ===========================================================================

class TestOllamaModelProperties:
    def test_model_type(self, ollama_model):
        model, _ = ollama_model
        assert model.model_type == "llava"

    def test_timeout_sec(self, ollama_model):
        model, _ = ollama_model
        assert model.timeout_sec == 30

    def test_pool_size(self, ollama_model):
        model, _ = ollama_model
        assert model.pool_size == 1

    def test_pool_initialised(self, ollama_model):
        model, _ = ollama_model
        assert len(model._pool) == 1
        assert "client" in model._pool[0]


class TestOllamaBasicSingleRequest:
    def _setup_response(self, mock_client, content=FAKE_RESPONSE_TEXT):
        mock_client.chat.return_value = {"message": {"content": content}}

    def test_returns_tuple_of_three(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        result = model.basic_single_request(user_prompt="hello")
        assert isinstance(result, tuple) and len(result) == 3

    def test_response_text_returned(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client, "my response")
        text, _, _ = model.basic_single_request(user_prompt="hello")
        assert text == "my response"

    def test_start_and_end_are_datetime_strings(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        _, start, end = model.basic_single_request(user_prompt="hello")
        fmt = "%Y-%m-%d %H:%M:%S"
        datetime.strptime(start, fmt)  # raises ValueError if format is wrong
        datetime.strptime(end, fmt)

    def test_user_prompt_included_in_messages(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="what is this?")
        call_kwargs = mock_client.chat.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any("what is this?" in m["content"] for m in user_msgs)

    def test_system_prompt_included_when_provided(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello", system_prompt="you are helpful")
        messages = mock_client.chat.call_args[1]["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "you are helpful"

    def test_no_system_prompt_by_default(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello")
        messages = mock_client.chat.call_args[1]["messages"]
        assert not any(m["role"] == "system" for m in messages)

    def test_temperature_passed_when_provided(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello", temperature=0.7)
        options = mock_client.chat.call_args[1]["options"]
        assert options["temperature"] == pytest.approx(0.7)

    def test_temperature_not_in_options_when_none(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello", temperature=None)
        options = mock_client.chat.call_args[1]["options"]
        assert "temperature" not in options

    def test_images_attached_to_user_message(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="describe", image_list=["base64data"])
        messages = mock_client.chat.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg.get("images") == ["base64data"]

    def test_empty_image_list_not_attached(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello", image_list=[])
        messages = mock_client.chat.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "images" not in user_msg

    def test_exception_returns_empty_string(self, ollama_model):
        model, mock_client = ollama_model
        mock_client.chat.side_effect = Exception("connection refused")
        text, start, end = model.basic_single_request(user_prompt="hello")
        assert text == ""
        assert start != "" and end != ""

    def test_exception_still_returns_timestamps(self, ollama_model):
        model, mock_client = ollama_model
        mock_client.chat.side_effect = Exception("timeout")
        _, start, end = model.basic_single_request(user_prompt="hello")
        fmt = "%Y-%m-%d %H:%M:%S"
        datetime.strptime(start, fmt)
        datetime.strptime(end, fmt)

    def test_model_type_passed_to_chat(self, ollama_model):
        model, mock_client = ollama_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello")
        assert mock_client.chat.call_args[1]["model"] == "llava"


# ===========================================================================
# OpenAIModel
# ===========================================================================

class TestOpenAIModelProperties:
    def test_model_type(self, openai_model):
        model, _ = openai_model
        assert model.model_type == "gpt-4.1"

    def test_pool_initialised(self, openai_model):
        model, _ = openai_model
        assert len(model._pool) == 1
        assert "client" in model._pool[0]


class TestOpenAIBasicSingleRequest:
    def _setup_response(self, mock_client, text=FAKE_RESPONSE_TEXT):
        mock_response = MagicMock()
        mock_response.output_text = text
        mock_client.responses.create.return_value = mock_response

    def test_returns_tuple_of_three(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        result = model.basic_single_request(user_prompt="hello")
        assert isinstance(result, tuple) and len(result) == 3

    def test_response_text_returned(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client, "openai answer")
        text, _, _ = model.basic_single_request(user_prompt="hello")
        assert text == "openai answer"

    def test_start_and_end_are_datetime_strings(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        _, start, end = model.basic_single_request(user_prompt="hello")
        fmt = "%Y-%m-%d %H:%M:%S"
        datetime.strptime(start, fmt)
        datetime.strptime(end, fmt)

    def test_user_prompt_in_request(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="what do you see?")
        call_kwargs = mock_client.responses.create.call_args[1]
        messages = call_kwargs["input"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any(
            any(c.get("text") == "what do you see?" for c in m["content"])
            for m in user_msgs
        )

    def test_system_prompt_included_when_provided(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello", system_prompt="be concise")
        messages = mock_client.responses.create.call_args[1]["input"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert any(c.get("text") == "be concise" for c in system_msgs[0]["content"])

    def test_no_system_prompt_by_default(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello")
        messages = mock_client.responses.create.call_args[1]["input"]
        assert not any(m["role"] == "system" for m in messages)

    def test_temperature_passed_when_provided(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello", temperature=0.5)
        assert mock_client.responses.create.call_args[1]["temperature"] == pytest.approx(0.5)

    def test_images_attached_to_user_content(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="describe", image_list=["b64img"])
        messages = mock_client.responses.create.call_args[1]["input"]
        user_msg = next(m for m in messages if m["role"] == "user")
        image_blocks = [c for c in user_msg["content"] if c.get("type") == "input_image"]
        assert len(image_blocks) == 1
        assert "b64img" in image_blocks[0]["image_url"]

    def test_multiple_images_all_attached(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="describe", image_list=["img1", "img2", "img3"])
        messages = mock_client.responses.create.call_args[1]["input"]
        user_msg = next(m for m in messages if m["role"] == "user")
        image_blocks = [c for c in user_msg["content"] if c.get("type") == "input_image"]
        assert len(image_blocks) == 3

    def test_empty_image_list_no_image_blocks(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello", image_list=[])
        messages = mock_client.responses.create.call_args[1]["input"]
        user_msg = next(m for m in messages if m["role"] == "user")
        image_blocks = [c for c in user_msg["content"] if c.get("type") == "input_image"]
        assert len(image_blocks) == 0

    def test_model_type_passed_to_create(self, openai_model):
        model, mock_client = openai_model
        self._setup_response(mock_client)
        model.basic_single_request(user_prompt="hello")
        assert mock_client.responses.create.call_args[1]["model"] == "gpt-4.1"