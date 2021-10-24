import argparse
import logging
import sys
import gurobipy as grb

import input_data
from branch_and_price import GAPBranchAndPrice
from standalone_model import \
    GAPStandaloneModelBuilder, \
    GAPStandaloneModelLpRelaxation, \
    DantzigWolfeFormulationGapStandaloneModelLpRelaxation
from standalone_model.dantzig_wolfe_formulation_gap_standalone_model_builder import \
    DantzigWolfeFormulationGapStandaloneModelBuilder


def main():
    try:
        fm_with_date = '%(asctime)s %(levelname)s: %(message)s'
        fmt_basic = '%(message)s'
        logging.basicConfig(format=fmt_basic,
                            datefmt='%Y/%m/%d %I:%M:%S %p',
                            level=logging.INFO)

        parser = argparse.ArgumentParser(description="Solves machine assignment problem.")
        # parser.add_argument('input_data',
        #                     help='Path to JSON file containing input data.')

        parser.add_argument('--method',
                            choices=['standalone', 'branch_and_price', 'both'],
                            default='both',
                            help='A method that should be used to solve a problem. default=both.')
        args = parser.parse_args()

        # solving facility problem
        # input_data = InputData.read(args.input_data)
        use_standalone_model = args.method in {'standalone', 'both'}
        use_branch_and_price = args.method in {'branch_and_price', 'both'}

        if use_standalone_model and False:
            # solve_using_standalone_model(input_data)
            gap = input_data.medium_example()
            # gap = input_data.example_applied_integer_programming()

            gap_model = GAPStandaloneModelBuilder(gap).build()
            gap_model.write()
            gap_model.solve()
            gap_model.report_results()

            lp_relaxation = GAPStandaloneModelLpRelaxation(mip_model=gap_model.mip_model)
            lp_relaxation.solve()
            lp_relaxation.report_results()

            dw_gap_model = DantzigWolfeFormulationGapStandaloneModelBuilder(gap_instance=gap).build()
            dw_gap_model.write()
            dw_gap_model.solve()
            dw_gap_model.report_results()

            dw_lp_relaxation = DantzigWolfeFormulationGapStandaloneModelLpRelaxation(mip_model=dw_gap_model.dw_model)
            dw_lp_relaxation.solve()
            dw_lp_relaxation.report_results()

        if use_branch_and_price:
            gap = input_data.medium_example()
            gap_model = GAPStandaloneModelBuilder(gap).build()
            gap_model.write()
            gap_model.solve()
            gap_model.report_results()

            GAPBranchAndPrice(gap).solve()

    except argparse.ArgumentError:
        logging.exception('Exception raised during parsing arguments')
        sys.exit(2)
    except grb.GurobiError:
        logging.exception("Gurobi exception thrown")
        sys.exit(1)
    except Exception:
        logging.exception("Exception occurred")
        sys.exit(1)


if __name__ == '__main__':
    main()

