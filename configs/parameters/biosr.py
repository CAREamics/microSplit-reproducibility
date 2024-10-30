from ._base import SplittingParameters


def biosr_musplit_parameters():
    return SplittingParameters().model_dump()

def biosr_denoisplit_parameters():
    return SplittingParameters().model_dump()