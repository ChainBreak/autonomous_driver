import pytest
from average_history_windows import RollingWindowChain, AverageHistoryWindows


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
    
   

