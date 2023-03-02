from enum import Enum


class DataType(Enum):
    NUM, TEXT, IMG, AUDIO = range(4)

    @classmethod
    def is_unstructured_except_text(cls, datatype):
        if datatype is DataType.NUM or datatype is DataType.TEXT:
            return False
        return True

    @classmethod
    def is_unstructured(cls, datatype):
        if datatype is DataType.NUM:
            return False
        return True