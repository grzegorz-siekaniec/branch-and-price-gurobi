# branch-and-price

Repository contains implementation of Branch-And-Price for Generalized Assignment Problem using Python and Gurobi solver.

Implementation is not intendent to be fast but rather descriptive.

See https://grzegorz-siekaniec.github.io/bits-of-this-bits-of-that/2021/solving-generalized-assignment-problem-using-branch-and-price.html for more details.

## How to run

In order to run application.

1. Create Python virtual environment:
   ```commandline
   python -m venv venv_branch_and_price

   ``` 
   Activate it - specifics depend on your operating system. For example on Linux with bash execute the following:
   ```commandline
   source ./venv_branch_and_price/bin/activate

   ``` 
   Install all requierements:
    ```commandline
   pip install -r /branch-and-price/src/requirements.txt

   ``` 
2. Go to directory `branch-and-price`.
3. Depending on method you want to choose to solve the problem, select different argument option:
    * to solve the problem using standalone model, execute:
        ```commandline
        python src/main.py --method standalone small_example

        ``` 
   
   * to solve the problem using Branch-And-Price, execute:
        ```commandline
        python src/main.py --method branch_and_price small_example

        ``` 
   
   * to solve the problem using both methods, execute::
        ```commandline
        python src/main.py small_example
        ```
   At the end you will also see a B&P tree and in working directory you will have plenty of LPs files containing every model that was solved.
