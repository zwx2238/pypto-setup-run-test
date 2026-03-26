source .env

python benchmark.py --category activation_functions loss_functions matrix_operations normalization other reduction --worker-mode local --pypto-run-mode 0 --parallel 16 --devices 0,1,2,3,4,5,6,7 --profile --llm adaptive_search --as-concurrent 2 --as-max-tasks 4

