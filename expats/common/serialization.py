
from abc import ABCMeta, abstractclassmethod, abstractmethod


class Serializable(metaclass=ABCMeta):

    @abstractclassmethod
    def load(cls, artifact_path: str) -> 'Serializable':
        """
        load serialized objects in-memory.
        Args:
            artifact_path: path to artifact (directory or file path)
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, artifact_path: str):
        """
        dump objects into the storage.
        Args:
            artifact_path: path to artifact (directory or file path)
        """
        raise NotImplementedError()
