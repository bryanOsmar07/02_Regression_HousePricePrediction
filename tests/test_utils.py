# tests/test_utils.py
from src.utils import load_object, save_object


def test_save_and_load_object(tmp_path):
    # Crear un objeto de prueba
    test_obj = {"x": 10, "y": 20}

    # Path temporal (pytest lo limpia luego)
    file_path = tmp_path / "obj.pkl"

    # Guardar objeto
    save_object(file_path, test_obj)

    # Cargar objeto
    loaded_obj = load_object(file_path)

    assert loaded_obj == test_obj
