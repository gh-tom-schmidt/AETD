import configparser


class Config:
    def __init__(self, file_name="debug_default.conf"):
        parser = configparser.ConfigParser()
        parser.read(file_name)

        # flatten the config sections into uppercase globals
        for section in parser.sections():
            for key, value in parser[section].items():
                # convert to int if possible
                if value.isdigit():
                    value = int(value)
                # store in class-level dictionary
                Config.data[key.upper()] = value

        # inject into module globals so they're directly accessible
        globals().update(Config.data)
