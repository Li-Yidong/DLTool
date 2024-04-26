"""app/utils/helper.py"""
import json
from enum import Enum
from typing import Callable

stringify_key_chain: Callable = lambda key_chain, start, end: ",".join(
    key_chain[start:end]
)
stringify_comment: Callable = lambda comment: "\n".join(comment)
dump_json: Callable = lambda var, indent: json.dumps(var, indent=indent)
print_json: Callable = lambda var, indent: print(dump_json(var, indent))


class InputTypes(Enum):
    PYQT = "pyqt"
    DICT = "Dict"
    PATH = "Path"
    BOOLEAN = "Boolean"
    DROPDOWN = "Dropdown"
    STR_INPUT = "StrInput"
    INT_INPUT = "IntInput"
    FLT_INPUT = "FltInput"
    SHP_INPUT = "ShpInput"
    SIZE_INPUT = "SizeInput"
    LIST_INPUT = "ListInput"
