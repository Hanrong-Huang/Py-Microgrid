from typing import Union

import PySAM.Singleowner as Singleowner

from py_microgrid.simulation.technologies.financial.custom_financial_model import CustomFinancialModel

FinancialModelType = Union[Singleowner.Singleowner, CustomFinancialModel]
