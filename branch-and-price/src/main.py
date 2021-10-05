import argparse
import logging
import sys
import gurobipy as grb

import input_data
from standalone_model.gap_standalone_model_builder import GAPStandaloneModelBuilder


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

        if use_standalone_model:
            # solve_using_standalone_model(input_data)
            gap = input_data.small_example()
            gap_model = GAPStandaloneModelBuilder(gap).build()
            gap_model.write()
            gap_model.solve()
            gap_model.report_results()

        if use_branch_and_price:
            # solve_using_benders_decomposition(input_data)
            pass

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

