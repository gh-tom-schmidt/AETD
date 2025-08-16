import configparser


class GenerateConfig:
    """
    Reads a .conf file and generates a global.pyi stub for Pylance.
    """

    def __init__(self, conf_file: str) -> None:
        parser = configparser.ConfigParser(interpolation=None)
        # keep original case of keys
        parser.optionxform = str
        parser.read(conf_file, encoding="utf-8")

        data = {}

        # include all sections
        for section in parser.sections():
            for key, value in parser[section].items():
                data[key.upper()] = self._convert_value(value)

        # generate the .pyi file
        with open("configs/global.pyi", "w", encoding="utf-8") as f:
            for key, value in data.items():
                if isinstance(value, int):
                    type_hint = "int"
                elif isinstance(value, float):
                    type_hint = "float"
                elif isinstance(value, bool):
                    type_hint = "bool"
                else:
                    type_hint = "str"
                f.write(f"{key}: {type_hint}\n")

    @staticmethod
    def _convert_value(value: str) -> str | int | float | bool:
        """Try to convert the string value to int, float, or bool."""
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            pass
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        return value


class Config:
    """
    Loads a .conf file and injects values into module globals().
    """

    data = {}

    def __init__(self, conf_file: str) -> None:
        parser = configparser.ConfigParser(interpolation=None)
        # keep original case of keys
        parser.optionxform = str
        parser.read(conf_file, encoding="utf-8")

        for section in parser.sections():
            for key, value in parser[section].items():
                # attempt type conversion
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                Config.data[key.upper()] = value

        # inject into module globals
        globals().update(Config.data)
