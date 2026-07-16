"""
Custom exceptions for the configuration package.
"""

from typing import Optional


class ConfigurationError(Exception):
    """
    Raised whenever the project configuration is invalid.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details

        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\n\nDetails:\n{self.details}"

        return self.message