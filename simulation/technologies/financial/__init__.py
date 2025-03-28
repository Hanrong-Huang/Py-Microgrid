from typing import Union

import PySAM.Singleowner as Singleowner

from Py_Microgrid.simulation.technologies.financial.custom_financial_model import CustomFinancialModel

FinancialModelType = Union[Singleowner.Singleowner, CustomFinancialModel]
