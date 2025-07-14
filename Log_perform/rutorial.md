## Data Comparison Chart

To visualize and compare training results between different models or maps, use the script `Log_perform/perform_chart.py`.

### Usage

1. Place your log files (e.g., `7_4_DDQN_TL.txt`, `5_1_DQN_TL.txt`, etc.) in the `Log_perform/` directory.

To filter the necessary values such as episode and reward, you need to add your log file to `log_filter.py` for filtering : 

```python
# filepath: Log_perform/log_filter.py
python log_filter.py
```

2. Open a terminal in this directory and run:
   
```python
# filepath: Log_perform/perform_chart.py
python perform_chart.py
```