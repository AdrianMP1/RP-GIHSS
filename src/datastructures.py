class MaxHeap:

    def __init__(self) -> None:
        
        self.size = 0
        
        self.hvc = []
        self.index = []
        self.solutions = []

    
    def add(self, sol, indx, hvc_value):
        """
        Add a new element to the heap
        """

        self.size += 1
        
        self.solutions.append(sol)
        self.index.append(indx)
        self.hvc.append(hvc_value)

        self._heapify_up(self.size - 1)


    def pop(self):
        """
        Remove and return the element with the largest HVC
        """

        if self.size == 0:
            return None
        
        # Swap the first and the last element
        self._swap(0, self.size - 1)
        
        # Remove the last element
        max_hvc = self.hvc.pop()
        max_indx = self.index.pop()
        best_sol = self.solutions.pop()
        
        # Update size
        self.size -= 1

        # Restore the heap property
        self._heapify_down(0)

        return best_sol


    def update(self, sol, new_hvc):
        """
        Update the HVC of an element and restore heap property
        """

        # Get the position
        pos = self.solutions.index(sol)
        
        # Update value
        self.hvc[pos] = new_hvc

        # Rebalance the heap
        self._heapify_up(pos)
        self._heapify_down(pos)
        


    def _heapify_up(self, pos):
        """
        Move a node up in the heap
        """

        parent = (pos - 1) // 2
        if pos > 0 and self.hvc[pos] > self.hvc[parent]:
            self._swap(pos, parent)
            self._heapify_up(parent)
    

    def _heapify_down(self, pos):
        """
        Move a node down in the heap
        """
        largest = pos
        left = 2 * pos + 1
        right = 2 * pos + 2

        if left < self.size and self.hvc[left] > self.hvc[largest]:
            largest = left
        
        if right < self.size and self.hvc[right] > self.hvc[largest]:
            largest = right

        if largest != pos:
            self._swap(pos, largest)
            self._heapify_down(largest)
    

    def _swap(self, i, j):
        """
        Swap two elements in the heap
        """
        self.hvc[i], self.hvc[j] = self.hvc[j], self.hvc[i]
        self.index[i], self.index[j] = self.index[j], self.index[i]
        self.solutions[i], self.solutions[j] = self.solutions[j], self.solutions[i]