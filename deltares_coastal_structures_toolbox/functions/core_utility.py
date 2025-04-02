# SPDX-License-Identifier: GPL-3.0-or-later
import warnings


def check_variable_validity_range(
    variable_name: str,
    formula_name: str,
    values: float,
    min_value: float,
    max_value: float,
) -> None:
    """Check if a variable is within the validity range for a formula and raise a warning if not.

    Parameters
    ----------
    variable_name : str
        Name of the variable to check.
    formula_name : str
        Name of the formula for which the validity range is checked.
    value : float
        Value of the variable to check.
    min_value : float
        Minimum value of the validity range.
    max_value : float
        Maximum value of the validity range.
    """
    if isinstance(values, (int, float)):
        values = [values]

    is_valid = []
    for value in values:
        if (value < min_value) or (value > max_value):
            warnings.warn(
                (
                    f"Value of {variable_name} ({value}) is outside of the validity range ({min_value} - {max_value})"
                    f" for the {formula_name} formula."
                )
            )
            is_valid.append(False)
        else:
            is_valid.append(True)

    return is_valid
