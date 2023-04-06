class InstanceParseException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InstanceGroundingException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
