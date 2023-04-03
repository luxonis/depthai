from pathlib import Path


def create_writer_dir(path: Path, name: str, extension: str = 'avi') -> str:
    if path.suffix == '':  # If path is a folder
        path.mkdir(parents=True, exist_ok=True)
        return str(path / f'{name}.{extension}')
    else:  # If path is a file
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
