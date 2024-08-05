import itertools
import backtest
import strategies

def grid_search(strategy_name, param_grid, data, symbol):
    """Perform a grid search over the parameter grid for a given strategy."""
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(*param_grid.values()))
    best_perf = float('-inf')
    best_outperf = float('-inf')
    best_params_perf = None
    best_params_outperf = None

    for param_values in param_combinations:
        # Create a dictionary of parameter names and values
        param_dict = dict(zip(param_grid.keys(), param_values))
        
        # Run the backtest
        perf, outperf, _ = backtest.run_backtest(
            strategy_class=strategies.__getattribute__(strategy_name),
            data=data,
            symbol=symbol,
            print_results=False,
            **param_dict
        )
        
        # Update the best performance based on perf
        if perf > best_perf:
            best_perf = perf
            best_params_perf = param_dict
        
        # Update the best performance based on outperf
        if outperf > best_outperf:
            best_outperf = outperf
            best_params_outperf = param_dict

    return {
        "best_perf": best_perf,
        "best_outperf": best_outperf,
        "best_params_perf": best_params_perf,
        "best_params_outperf": best_params_outperf
    }
