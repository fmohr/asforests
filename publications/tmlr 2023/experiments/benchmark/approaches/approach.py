from abc import ABC, abstractmethod


class Approach(ABC):

    @property
    @abstractmethod
    def deviation_means(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def deviation_vars(self):
        raise NotImplementedError

    def update(self, deviation_matrix):
        raise NotImplementedError
