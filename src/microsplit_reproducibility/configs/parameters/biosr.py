from ._base import SplittingParameters


def get_musplit_parameters() -> dict:
    return SplittingParameters().model_dump()

def get_denoisplit_parameters() -> dict:
    return SplittingParameters().model_dump()