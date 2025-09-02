from  configparser import ConfigParser


class Config:
    def __init__(self, config_file="./src/dataingestion/fileconfig.ini"):
        self.config = ConfigParser()
        # print(self.config)
        self.config.read(config_file)

    def get_state_file(self):
        return self.config["DEFAULT"].get("STATE_FILE")

    def get_watch_folder(self):
        # dir = self.config["DEFAULT"].get("WATCH_FOLDER")
        # print(dir)
        return self.config["DEFAULT"].get("WATCH_FOLDER")

