import pytest
from history_digest import RollingWindowChain, HistoryDigest
import numpy as np

@pytest.mark.parametrize("window_size, values, expected_average", [
    (1, [1, 2, 3, 4], 4),
    (2, [1, 2, 3, 4], 3.5),
    (3, [1, 2, 3, 4], 3),
    (4, [1, 2, 3, 4], 2.5),
    (5, [1, 2, 3, 4], 2.5),
])
def test_rolling_window_chain_basic(window_size, values, expected_average):
    """Test basic RollingWindowChain functionality with different window sizes"""
    # Test single window
    window = RollingWindowChain(window_size)

    for value in values:
        window.push(value)

    assert window.get_average() == expected_average
    
   
@pytest.mark.parametrize("window_sizes, values", [
    ([1, 2], [1, 2, 3, 4, 5, 6]),
    ([2, 3], [1, 2, 3, 4, 5, 6]),
    ([3, 2], [1, 2, 3, 4, 5, 6]),
    ([4, 1], [1, 2, 3, 4, 5, 6]),
    ([5, 1], [1, 2, 3, 4, 5, 6]),
    ([1, 2], np.array([[i,2*i] for i in range(7)])),
    ([3, 2], np.array([[i,2*i] for i in range(7)])),
])
def test_average_history_windows(window_sizes, values):
    """Test HistoryDigest with multiple window sizes"""
    # Create windows of size 2 and 4
    windows = HistoryDigest(window_sizes)

    for value in values:
        windows.push(value)

    window_averages =  windows.get_window_averages()

    values_reversed = values[::-1]
    i1 = 0
    i2 = 0
    for window_size, window_average in zip(window_sizes, window_averages):
        i2 = i1 + window_size
        assert np.all(window_average == np.mean(values_reversed[i1:i2],axis=0))
        i1 = i2
    
@pytest.mark.parametrize("num_windows, growth_rate, expected_window_sizes", [
    (4, 2.0, [1, 2, 4, 8]),
    (4, 1.5, [1, 1, 2, 3]),
    (5, 1.2, [1, 1, 1, 1, 2]),
])
def test_average_history_windows_from_window_growth_rate(num_windows, growth_rate, expected_window_sizes):
    windows = HistoryDigest.from_window_growth_rate(num_windows=num_windows, growth_rate=growth_rate)
    for window, expected_window_size in zip(windows.windows, expected_window_sizes):
        assert window.window_size == expected_window_size
    

if __name__ == "__main__":
    pytest.main([__file__]) 