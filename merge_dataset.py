import os
from pathlib import Path

# Specify the directory
left_directory = Path('dataset_cam7_may_18_part2/left')
right_directory = Path('dataset_cam7_may_18_part2/right')
vertical_directory = Path('dataset_cam7_may_18_part2/vertical')


directories = [left_directory, right_directory, vertical_directory]

for directory in directories:
    # Loop over the files
    for index, file in enumerate(sorted(directory.iterdir()), start=1):
        # Separate the stem and suffix of the file
        stem, suffix = file.stem, file.suffix
        print(f'Stem: {stem}, suffix: {suffix}')
        # Create the new filename with the index at the end
        new_filename = directory / f'{stem}_{index}{suffix}'
        # Rename the file
        file.rename(new_filename)

