from enum import Enum


class BaseEnum(Enum):

    def __str__(self):
        return self.name

    @classmethod
    def get_elements_list(cls):
        return [e for e in cls]
