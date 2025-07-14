## Data Comparison Chart

To visualize and compare training results between different models or maps, use the script `Log_perform/perform_chart.py`.

### Usage

Place your log files (e.g., `7_4_DDQN_TL.txt`, `5_1_DQN_TL.txt`, etc.) in the `Log_perform/` directory.

To filter the necessary values such as episode and reward, you need to add your log file to `log_filter.py` for filtering : 

```python
python log_filter.py
```

2. Open a terminal in this directory and run:
   
```python
python perform_chart.py
```