from collections import deque


class AverageHistoryWindows:
    def __init__(self, window_sizes:list[int]):

        self.windows = self._build_window_chain(window_sizes)

    def _build_window_chain(self,window_sizes:list[int])->list["RollingWindowChain"]:
        windows:list["RollingWindowChain"] = []

        for window_size in window_sizes:
            new_window = RollingWindowChain(window_size)
            # Link the new window to the previous window if it exists
            if len(windows) > 0:
                windows[-1].link_next_window(new_window)

            windows.append(new_window)

        return windows
    
    def get_average(self):
        return [window.get_average() for window in self.windows]
    
    def push(self, value):
        # Add the value to the first window. Everything else is handled by the chain
        self.windows[0].push(value)

class RollingWindowChain:
    """
    A rolling window that calculates the average of the values in the window.
    When the window is full, the oldest value is popped and push to the next window.
    """
    def __init__(self, window_size:int):
        self.window_size = window_size
        self.next_window = None
        self.queue = deque()
        self.sum = 0

    def link_next_window(self, next_window:"RollingWindowChain"):
        self.next_window = next_window

    def push(self, value):
        """
        push the window with a new value. Pop and return the oldest value if the window is full.
        """
        self.queue.appendleft(value)

        # add the value to the sum
        self.sum += value
        
        if len(self.queue) > self.window_size:

            pop_value = self.queue.pop()

            # remove the value from the sum
            self.sum -= pop_value
            
            # Pass the oldest value to the next window if it exists
            if self.next_window is not None:
                self.next_window.push(value=pop_value)
            
    def get_average(self):
        return self.sum / len(self.queue)
    
    def get_window(self):
        return list(self.queue)