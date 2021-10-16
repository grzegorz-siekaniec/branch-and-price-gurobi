import logging

import gurobipy as grb

from common import is_non_zero


class DantzigWolfeFormulationGapStandaloneModelLpRelaxation:

    def __init__(self,
                 mip_model: grb.Model):
        self._lp_relaxation = mip_model.relax()

    def solve(self):
        self._lp_relaxation.Params.LogToConsole = 0
        self._lp_relaxation.optimize()

    def write(self):
        model_name = self._lp_relaxation.getAttr(grb.GRB.Attr.ModelName)
        self._lp_relaxation.write(f'{model_name}.lp')

    def report_results(self):
        obj_val = self._lp_relaxation.getAttr(grb.GRB.Attr.ObjVal)

        logging.info("** Final results using LP relaxation of Dantzig-Wolfe formulation of standalone model! **")
        logging.info("Objective value: %f", obj_val)

        for var in self._lp_relaxation.getVars():
            if is_non_zero(var.x):
                logging.info(f'{var.VarName} \t:{var.X}')

        logging.info('')
