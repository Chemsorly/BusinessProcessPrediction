class Error(Exception):
    pass

class ConducthorError(Error):
    """Exception raised for errors during execution. Handled via custom element to make parsing easier

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message