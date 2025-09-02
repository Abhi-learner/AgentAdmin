import pandas as pd
from typing import Dict, Any, List, Tuple
from src.state.diskalertstate import DiskAlertState

class StatePrinter:
    def __init__(self, state: DiskAlertState):
        self.state = state

    def _print_dict(self, d: Dict[str, Any]):
        df = pd.DataFrame(list(d.items()), columns=["Key", "Value"])
        print(df.to_string(index=False))

    def _print_list(self, lst: List[Any], key: str):
        if all(isinstance(v, dict) for v in lst):
            df = pd.DataFrame(lst)
            print(df.to_string(index=False))
        else:
            df = pd.DataFrame(lst, columns=[key])
            print(df.to_string(index=False))

    def print_state(self):
        for key, value in self.state.items():
            print(f"\n=== {key.upper()} ===")
            if isinstance(value, dict):
                self._print_dict(value)
            elif isinstance(value, list):
                self._print_list(value, key)
            else:
                print(value)
