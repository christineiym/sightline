"""
Interfaces to dataset Generators, which create images with specified characteristics.
"""
import itertools
from abc import ABC, abstractmethod
from csv import DictWriter, DictReader
from sightline.characteristics import Characteristic


FIELD_DEFAULT = "default"
FIELD_SYSTEMATIC = "systematic"
FIELD_RANDOM = "random"

# Assumption: You can take care of dataset-wide random characteristics outside Generator, and add to values with defaults
SCOPE_IMAGE = "image"
SCOPE_OBJECT = "object"  # Object metadata (shape, position, color, size)


class Generator(ABC):
    def generate_dataset(self, out_dir: str, out_csv: str, pair: bool, repetitions_per_combination, characteristics, step, n=None):
        """Generate metadata for n images and write CSV. CSV has one row per dot."""
        metadata: list[dict] = self.generate_image_metadata(out_dir=out_dir, pair=pair, \
            repetitions_per_combination=repetitions_per_combination, characteristics=characteristics, n=n, step=step)

        # write CSV
        with open(out_csv, "w", newline="") as f:
            writer = DictWriter(f, fieldnames=metadata[0].keys())
            writer.writeheader()
            for row in metadata:
                writer.writerow(row)
        
        print(f"Generated images in {out_dir}, CSV: {out_csv}")
        return out_csv

    def generate_image_metadata(self, out_dir: str, pair: bool, repetitions_per_combination, characteristics: dict[str, list[Characteristic]], step, n=None) -> list[dict]:
        """Generate metadata for images in the set."""
        # Unpack and set dataset-wide constants.
        default_values = {}
        if FIELD_DEFAULT in characteristics.keys():
            for item in characteristics[FIELD_DEFAULT]:
                # print({item.name, item.default})
                default_values.update({item.name: item.default})

        # Variables of interest
        ## The systematic ones
        systematic_values = {}
        for index, item in enumerate(characteristics[FIELD_SYSTEMATIC]):
            actual_n = None if n is None else n[index]
            categories = item.get_systematic_sample(n_max=actual_n, step=step[index])
            systematic_values.update({item.name: list(categories)})
        names = list(systematic_values.keys())
        value_lists = list(systematic_values.values())
        print(systematic_values)
        systematic_values_flattened = [dict(zip(names, item)) for item in itertools.product(*value_lists)]  # cross product along category values
        # Future: What about stratified random sampling?

        ## The randomized ones where there is one value per image
        expected_total = len(systematic_values_flattened) * repetitions_per_combination
        random_values = {}
        other_random_values = []
        for item in characteristics[FIELD_RANDOM]:
            if item.scope == SCOPE_IMAGE:
                random_values.update({item.name: item.get_random_sample(expected_total)})
            else:
                other_random_values.append(item)

        current_index = 0  # future: validate against expected_total. Better indexing?
        all_rows = []
        for test_combination in systematic_values_flattened:
            i = 0
            while i < repetitions_per_combination:
                current_info = {  
                    "img_index": current_index,
                    "base_filename": f"{current_index:05d}",
                    "images": f"{current_index:05d}_a.png|{current_index:05d}_b.png",
                    **default_values,  # technically this could be moved outside the loop
                    **random_values,  # also this
                    **test_combination
                }

                # Generate image contents as rows
                row_info, spare_characteristics = self.generate_contents(pair=pair, 
                    constants=current_info, random_characteristics=other_random_values)
                
                # Generate image pair from its rows' metadata
                self.generate_image(out_dir=out_dir, pair=pair, object_info=row_info, **current_info)
                
                # Collect info for the output file
                all_rows.extend(row_info)
                
                i += 1
                current_index += 1

        return all_rows
    
    def generate_images_from_file(self, input_csv: str, out_dir: str, pair: bool, id_column: str):
        grouped_data = {}
        with open(input_csv, mode='r', newline='') as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                id = row[id_column]
                grouped_data.setdefault(id, []).append(row)
        
        for image in grouped_data:
            self.generate_image(out_dir=out_dir, pair=pair, object_info=grouped_data[image], **grouped_data[image][0])
        
        print(f"Generated images in {out_dir}")
        return grouped_data

    @abstractmethod
    def generate_contents(self, *args, **kwargs) -> tuple[list[dict], list[Characteristic]]:
        """Generate metadata that changes between images in the set."""
        pass

    @abstractmethod
    def generate_image(self, index, out_dir: str, *args, **kwargs):
        """Generate one image pair based on provided metadata and return path."""
        pass